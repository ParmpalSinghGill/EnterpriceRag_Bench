"""Generate BAAI/bge-m3 dense embeddings and save to Postgres chunks.embedding.

Usage (prefer conda env with torch + transformers, e.g. LLM):
  conda run -n LLM python -m ingestion.embed_bge_m3 --build-id chunk_v1

Resumable: only rows with embedding IS NULL are processed.
Reconnects automatically if Postgres drops the connection mid-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
import torch
import torch.nn.functional as F
from pgvector.psycopg import register_vector
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "BAAI/bge-m3"
DIM = 1024
C = 8192

_LOCAL_SNAPS = [
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/9a0624b896d81da7492a910ffa53731274b6cf3d",
]


def resolve_model_path(explicit: str | None) -> str:
    if explicit and Path(explicit).exists():
        return explicit
    env = os.environ.get("ERAG_BGE_M3_PATH")
    if env and Path(env).exists():
        return env
    for p in _LOCAL_SNAPS:
        if (p / "config.json").exists() and (
            (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists()
        ):
            return str(p)
    return MODEL_ID


def connect(dsn: str):
    conn = psycopg.connect(
        dsn,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        autocommit=False,
    )
    register_vector(conn)
    conn.execute("SET synchronous_commit = off")
    return conn


def load_model(model_path: str, device: str, dtype: torch.dtype):
    print(f"loading model from {model_path} → {device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    model.to(device=device, dtype=dtype)
    return tok, model


@torch.inference_mode()
def embed_batch(tok, model, texts: list[str], device: str, dtype: torch.dtype) -> np.ndarray:
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=C,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    emb = out.last_hidden_state[:, 0].float()
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


def choose_batch_size(n_tokens: int, base: int) -> int:
    if n_tokens > 4096:
        return 1
    if n_tokens > 2048:
        return max(1, base // 8)
    if n_tokens > 1024:
        return max(1, base // 4)
    if n_tokens > 512:
        return max(1, base // 2)
    return base


def _alive(conn) -> bool:
    try:
        if conn is None or conn.closed:
            return False
        conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _reconnect(dsn: str, old=None):
    if old is not None:
        try:
            old.close()
        except Exception:
            pass
    for attempt in range(1, 8):
        try:
            conn = connect(dsn)
            print(f"  reconnected to postgres (attempt {attempt})")
            return conn
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f"  reconnect failed ({e}); sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError("could not reconnect to postgres")


def run_embed(
    dsn: str,
    build_id: str,
    model_path: str,
    batch_size: int,
    device: str,
    fp16: bool,
    create_index: bool,
    limit: int | None,
):
    dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
    tok, model = load_model(model_path, device, dtype)

    t0 = time.time()
    conn = connect(dsn)

    def counts():
        nonlocal conn
        if not _alive(conn):
            conn = _reconnect(dsn, conn)
        row = conn.execute(
            """
            SELECT COUNT(*) FILTER (WHERE embedding IS NULL),
                   COUNT(*) FILTER (WHERE embedding IS NOT NULL),
                   COUNT(*)
            FROM chunks WHERE build_id=%s
            """,
            (build_id,),
        ).fetchone()
        return row

    n_null, n_filled, n_total = counts()
    print(f"build={build_id} total={n_total} filled={n_filled} remaining={n_null}")

    update_sql = """
        UPDATE chunks SET embedding = %s::vector
        WHERE build_id=%s AND chunk_id=%s
    """
    fetch_sql = """
        SELECT chunk_id, text, n_tokens
        FROM chunks
        WHERE build_id=%s AND embedding IS NULL
        ORDER BY n_tokens ASC, chunk_id
        LIMIT %s
    """

    if n_null == 0:
        print("nothing to embed")
    else:
        target = n_null if limit is None else min(n_null, int(limit))
        print(f"embedding up to {target} chunks (streamed, resumable)…")
        WINDOW = 1024
        n_done = 0
        while n_done < target:
            take = min(WINDOW, target - n_done)
            try:
                if not _alive(conn):
                    conn = _reconnect(dsn, conn)
                pending = conn.execute(fetch_sql, (build_id, take)).fetchall()
            except psycopg.OperationalError as e:
                print(f"  fetch lost connection: {e}")
                conn = _reconnect(dsn, conn)
                pending = conn.execute(fetch_sql, (build_id, take)).fetchall()
            if not pending:
                break

            i = 0
            while i < len(pending):
                nt0 = int(pending[i][2] or 0)
                bs = choose_batch_size(nt0, batch_size)
                batch = pending[i : i + bs]
                max_nt = max(int(r[2] or 0) for r in batch)
                bs = min(len(batch), choose_batch_size(max_nt, batch_size))
                batch = pending[i : i + bs]
                ids = [r[0] for r in batch]
                texts = [r[1] or "" for r in batch]

                try:
                    embs = embed_batch(tok, model, texts, device, dtype)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(batch) == 1:
                        raise
                    print(f"  OOM at batch={len(batch)} n_tokens≈{max_nt}; retrying singly")
                    embs_list = [
                        embed_batch(tok, model, [r[1] or ""], device, dtype)[0]
                        for r in batch
                    ]
                    embs = np.stack(embs_list, axis=0)

                assert embs.shape[1] == DIM, embs.shape
                rows = [(embs[j].tolist(), build_id, ids[j]) for j in range(len(ids))]

                for attempt in range(1, 6):
                    try:
                        if not _alive(conn):
                            conn = _reconnect(dsn, conn)
                        with conn.cursor() as ucur:
                            ucur.executemany(update_sql, rows)
                        conn.commit()
                        break
                    except psycopg.OperationalError as e:
                        print(f"  write lost connection (attempt {attempt}): {e}")
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        conn = _reconnect(dsn, conn)
                        if attempt == 5:
                            raise

                n_done += len(batch)
                i += len(batch)
                if n_done % 500 < len(batch) or n_done >= target:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-6)
                    eta = (target - n_done) / max(rate, 1e-6)
                    print(
                        f"  … {n_done}/{target} "
                        f"({rate:.1f} chunks/s, elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m) "
                        f"last_bs={len(batch)} max_tok≈{max_nt}"
                    )
                    if device.startswith("cuda"):
                        print(
                            f"     vram_alloc={torch.cuda.memory_allocated()/1e9:.2f}G "
                            f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}G"
                        )

    n_null, n_filled, n_total = counts()
    print(f"verify: total={n_total} filled={n_filled} null={n_null}")

    sample = conn.execute(
        """
        SELECT vector_dims(embedding) FROM chunks
        WHERE build_id=%s AND embedding IS NOT NULL LIMIT 5
        """,
        (build_id,),
    ).fetchall()
    dims = [r[0] for r in sample]
    print(f"sample vector_dims={dims}")

    if create_index and n_null == 0 and n_filled > 0:
        print("creating HNSW index on chunks.embedding (cosine)…")
        conn.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
        t_idx = time.time()
        conn.execute(
            """
            CREATE INDEX chunks_embedding_hnsw_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        conn.commit()
        print(f"HNSW index created in {time.time()-t_idx:.0f}s")

    conn.execute(
        """
        UPDATE ingest_builds
        SET notes = %s
        WHERE build_id=%s
        """,
        (
            f"bge-m3 embeddings filled={n_filled}/{n_total}; dim={DIM}; fp16={fp16}",
            build_id,
        ),
    )
    conn.commit()
    conn.close()

    man_path = ROOT / "indexes" / build_id / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        man["embedder"] = {"name": MODEL_ID, "dim": DIM, "pooling": "cls", "normalize": True}
        man["phase"] = "embeddings_filled" if n_null == 0 else "embeddings_partial"
        man["n_embeddings"] = n_filled
        man["n_embeddings_null"] = n_null
        man_path.write_text(json.dumps(man, indent=2) + "\n")
        print(f"updated {man_path}")

    report = {
        "build_id": build_id,
        "model": MODEL_ID,
        "dim": DIM,
        "device": device,
        "fp16": fp16,
        "n_total": n_total,
        "n_filled": n_filled,
        "n_null": n_null,
        "elapsed_sec": round(time.time() - t0, 1),
        "create_index": create_index and n_null == 0,
    }
    out = ROOT / "indexes" / build_id / "embed_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description="Embed chunks with BAAI/bge-m3 → Postgres")
    ap.add_argument("--dsn", default=os.environ.get("ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"))
    ap.add_argument("--build-id", default="chunk_v1")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--create-index", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model_path)
    run_embed(
        dsn=args.dsn,
        build_id=args.build_id,
        model_path=model_path,
        batch_size=args.batch_size,
        device=args.device,
        fp16=args.fp16,
        create_index=args.create_index,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
