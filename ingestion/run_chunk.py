"""Chunk the EnterpriseRAG-Bench corpus per Docs/ingestion_spec.md (no embeddings).

Examples:
  # Size dry-run on full corpus (approx tokens, fast)
  python -m ingestion.run_chunk --token-mode bge_m3 --out indexes/chunk_v1

  # Small sample
  python -m ingestion.run_chunk --limit 1000 --out indexes/chunk_sample

  # Load into Docker Postgres after chunking
  python -m ingestion.run_chunk --token-mode bge_m3 --out indexes/chunk_v1 --load-postgres
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# Allow `python -m ingestion.run_chunk` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.chunker import (
    ChunkerConfig,
    DocRecord,
    chunk_document,
    merge_duplicate_docs,
)
from ingestion.tokenize import get_counter

DEFAULT_PARQUET = os.environ.get(
    "ERAG_DOCUMENTS_PARQUET",
    str(
        Path.home()
        / ".cache/huggingface/hub/datasets--onyx-dot-app--EnterpriseRAG-Bench"
    ),
)


def find_parquet(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    env = os.environ.get("ERAG_DOCUMENTS_PARQUET")
    if env and Path(env).exists():
        return Path(env)
    # Common local cache locations
    candidates = [
        Path("/media/parmpal/Data/DataSet/Cache/huggingface/hub/datasets--onyx-dot-app--EnterpriseRAG-Bench"),
        Path.home() / ".cache/huggingface/hub/datasets--onyx-dot-app--EnterpriseRAG-Bench",
    ]
    for base in candidates:
        if not base.exists():
            continue
        hits = list(base.glob("snapshots/*/data/documents/test.parquet"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "Could not find documents parquet. Pass --parquet PATH or set ERAG_DOCUMENTS_PARQUET."
    )


def iter_docs(parquet_path: Path, limit: int | None):
    pf = pq.ParquetFile(parquet_path)
    n = 0
    for batch in pf.iter_batches(
        batch_size=2000, columns=["doc_id", "source_type", "title", "content"]
    ):
        d = batch.to_pydict()
        for i in range(len(d["doc_id"])):
            yield DocRecord(
                doc_id=d["doc_id"][i],
                source_type=d["source_type"][i],
                title=d["title"][i] or "",
                content=d["content"][i] or "",
            )
            n += 1
            if limit is not None and n >= limit:
                return


def percentile(sorted_vals, p: float):
    if not sorted_vals:
        return 0
    idx = int((len(sorted_vals) - 1) * p)
    return sorted_vals[idx]


def run_chunk(
    parquet: Path,
    out_dir: Path,
    token_mode: str,
    limit: int | None,
    write_jsonl: bool,
    build_id: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = get_counter(token_mode)
    cfg = ChunkerConfig()

    print(f"parquet: {parquet}")
    print(f"token_mode={token_mode} C={cfg.C} T={cfg.T} O={cfg.O} pack={cfg.pack}")
    print("loading + merging duplicate doc_ids…")
    t0 = time.time()
    docs = merge_duplicate_docs(iter_docs(parquet, limit))
    print(f"docs after merge: {len(docs)} ({time.time() - t0:.1f}s)")

    # stats
    n_chunks = 0
    chars_chunks = 0
    chars_docs = 0
    tok_list = []
    by_source_chunks = Counter()
    by_source_docs = Counter()
    multi_chunk_docs = Counter()
    max_tok = 0
    over_C = 0
    map_path = out_dir / "docid_map.jsonl"
    chunks_path = out_dir / "chunks.jsonl"

    map_f = open(map_path, "w", encoding="utf-8")
    chunks_f = open(chunks_path, "w", encoding="utf-8") if write_jsonl else None

    t1 = time.time()
    for i, doc in enumerate(docs):
        chars_docs += len(doc.content or "")
        by_source_docs[doc.source_type] += 1
        chunks = chunk_document(doc, counter, cfg)
        if len(chunks) > 1:
            multi_chunk_docs[doc.source_type] += 1
        for ch in chunks:
            n_chunks += 1
            chars_chunks += ch.n_chars
            tok_list.append(ch.n_tokens)
            by_source_chunks[doc.source_type] += 1
            max_tok = max(max_tok, ch.n_tokens)
            if ch.n_tokens > cfg.C:
                over_C += 1
            map_f.write(
                json.dumps(
                    {
                        "chunk_id": ch.chunk_id,
                        "doc_id": ch.doc_id,
                        "source_type": ch.source_type,
                        "chunk_index": ch.chunk_index,
                        "n_tokens": ch.n_tokens,
                        "n_chars": ch.n_chars,
                        "region": ch.region,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if chunks_f:
                chunks_f.write(
                    json.dumps(
                        {
                            "chunk_id": ch.chunk_id,
                            "doc_id": ch.doc_id,
                            "source_type": ch.source_type,
                            "chunk_index": ch.chunk_index,
                            "region": ch.region,
                            "n_tokens": ch.n_tokens,
                            "n_chars": ch.n_chars,
                            "header": ch.header,
                            "body": ch.body,
                            "text": ch.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if (i + 1) % 20000 == 0:
            print(f"  … {i+1}/{len(docs)} docs, {n_chunks} chunks ({time.time()-t1:.1f}s)")

    map_f.close()
    if chunks_f:
        chunks_f.close()

    tok_list.sort()
    report = {
        "build_id": build_id,
        "pack": cfg.pack,
        "token_mode": token_mode,
        "C": cfg.C,
        "T": cfg.T,
        "O": cfg.O,
        "n_docs": len(docs),
        "n_chunks": n_chunks,
        "chunks_per_doc_mean": round(n_chunks / max(len(docs), 1), 4),
        "doc_content_chars": chars_docs,
        "doc_content_bytes_utf8_approx": chars_docs,  # ascii-heavy; see jsonl sizes for true
        "chunk_text_chars": chars_chunks,
        "token_stats": {
            "median": percentile(tok_list, 0.5),
            "p90": percentile(tok_list, 0.9),
            "p95": percentile(tok_list, 0.95),
            "p99": percentile(tok_list, 0.99),
            "max": max_tok,
            "over_C": over_C,
        },
        "by_source": {
            st: {
                "docs": by_source_docs[st],
                "chunks": by_source_chunks[st],
                "multi_chunk_docs": multi_chunk_docs[st],
                "multi_chunk_pct": round(
                    100.0 * multi_chunk_docs[st] / max(by_source_docs[st], 1), 2
                ),
                "chunks_per_doc": round(
                    by_source_chunks[st] / max(by_source_docs[st], 1), 3
                ),
            }
            for st in sorted(by_source_docs)
        },
        "artifacts": {
            "docid_map": str(map_path),
            "chunks_jsonl": str(chunks_path) if write_jsonl else None,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # On-disk sizes
    def fsize(p: Path) -> int:
        return p.stat().st_size if p.exists() else 0

    report["on_disk_bytes"] = {
        "docid_map_jsonl": fsize(map_path),
        "chunks_jsonl": fsize(chunks_path) if write_jsonl else 0,
    }
    report["on_disk_human"] = {
        k: _human(v) for k, v in report["on_disk_bytes"].items()
    }

    report_path = out_dir / "size_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "dataset": "onyx-dot-app/EnterpriseRAG-Bench",
        "config": "documents",
        "split": "test",
        "comparable": limit is None,
        "n_docs_expected": 511962 if limit is None else None,
        "n_docs_indexed": len(docs),
        "n_chunks": n_chunks,
        "pack": cfg.pack,
        "C": cfg.C,
        "T": cfg.T,
        "O": cfg.O,
        "embedder": None,
        "sparse": None,
        "token_mode": token_mode,
        "title_header": True,
        "phase": "chunk_only_no_embeddings",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== SIZE REPORT ===")
    print(json.dumps({k: report[k] for k in [
        "n_docs", "n_chunks", "chunks_per_doc_mean", "doc_content_chars",
        "chunk_text_chars", "token_stats", "on_disk_human", "elapsed_sec"
    ]}, indent=2))
    print(f"wrote {report_path}")
    return report


def _human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def load_postgres(out_dir: Path, build_id: str, dsn: str, load_docs: bool = True):
    """Stream chunks.jsonl into Postgres (phase 1: text only, embedding NULL)."""
    try:
        import psycopg
    except ImportError as e:
        raise SystemExit("psycopg is required for --load-postgres: pip install psycopg[binary]") from e

    chunks_path = out_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit("chunks.jsonl missing — re-run with --write-jsonl (default on)")

    manifest = json.loads((out_dir / "manifest.json").read_text())
    report = json.loads((out_dir / "size_report.json").read_text())

    insert_chunk_sql = """
        INSERT INTO chunks (
            chunk_id, build_id, doc_id, source_type, chunk_index,
            region, n_tokens, n_chars, header, body, text
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (build_id, doc_id, chunk_index) DO UPDATE SET
            chunk_id = EXCLUDED.chunk_id,
            body = EXCLUDED.body,
            text = EXCLUDED.text,
            n_tokens = EXCLUDED.n_tokens,
            n_chars = EXCLUDED.n_chars,
            region = EXCLUDED.region,
            header = EXCLUDED.header,
            source_type = EXCLUDED.source_type
    """
    insert_doc_sql = """
        INSERT INTO documents (doc_id, source_type, title, content, content_chars)
        VALUES (%s,%s,%s,'',0)
        ON CONFLICT (doc_id) DO NOTHING
    """

    t0 = time.time()
    n_chunks = 0
    n_docs = 0

    with psycopg.connect(dsn) as conn:
        conn.execute("SET synchronous_commit = off")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingest_builds (
                    build_id, pack, c_tokens, t_tokens, o_tokens, token_mode,
                    comparable, n_docs, n_chunks, notes
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (build_id) DO UPDATE SET
                    n_docs=EXCLUDED.n_docs,
                    n_chunks=EXCLUDED.n_chunks,
                    notes=EXCLUDED.notes
                """,
                (
                    build_id,
                    manifest["pack"],
                    manifest["C"],
                    manifest["T"],
                    manifest["O"],
                    manifest["token_mode"],
                    manifest.get("comparable", True),
                    report["n_docs"],
                    report["n_chunks"],
                    "phase1 chunk-only; embeddings NULL",
                ),
            )
            cur.execute("DELETE FROM chunks WHERE build_id=%s", (build_id,))
        conn.commit()

        # Pass 1: documents (unique doc_ids)
        if load_docs:
            print("pass 1/2: upserting documents…")
            seen: set[str] = set()
            doc_batch = []
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    doc_id = row["doc_id"]
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    header = row["header"]
                    title = header.split("]", 1)[-1].strip() if "]" in header else header
                    doc_batch.append((doc_id, row["source_type"], title))
                    if len(doc_batch) >= 2000:
                        with conn.cursor() as cur:
                            cur.executemany(insert_doc_sql, doc_batch)
                        conn.commit()
                        n_docs += len(doc_batch)
                        doc_batch.clear()
                        if n_docs % 50000 == 0:
                            print(f"  … {n_docs} docs ({time.time()-t0:.0f}s)")
            if doc_batch:
                with conn.cursor() as cur:
                    cur.executemany(insert_doc_sql, doc_batch)
                conn.commit()
                n_docs += len(doc_batch)
            print(f"  documents upserted: {n_docs}")

        # Pass 2: chunks
        print("pass 2/2: inserting chunks…")
        batch = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                batch.append(
                    (
                        f"{build_id}:{row['chunk_id']}",
                        build_id,
                        row["doc_id"],
                        row["source_type"],
                        row["chunk_index"],
                        row.get("region"),
                        row["n_tokens"],
                        row["n_chars"],
                        row["header"],
                        row["body"],
                        row["text"],
                    )
                )
                if len(batch) >= 500:
                    with conn.cursor() as cur:
                        cur.executemany(insert_chunk_sql, batch)
                    conn.commit()
                    n_chunks += len(batch)
                    batch.clear()
                    if n_chunks % 25000 == 0:
                        print(f"  … {n_chunks} chunks ({time.time()-t0:.0f}s)")
        if batch:
            with conn.cursor() as cur:
                cur.executemany(insert_chunk_sql, batch)
            conn.commit()
            n_chunks += len(batch)

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ingest_builds SET n_docs=%s, n_chunks=%s WHERE build_id=%s",
                (n_docs or report["n_docs"], n_chunks, build_id),
            )
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE build_id=%s",
                (build_id,),
            )
            db_n = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FILTER (WHERE embedding IS NULL) FROM chunks WHERE build_id=%s",
                (build_id,),
            )
            null_emb = cur.fetchone()[0]
        conn.commit()

    print(
        f"loaded build_id={build_id} into postgres: "
        f"{n_chunks} chunks, {n_docs} docs, db_count={db_n}, null_embeddings={null_emb} "
        f"({time.time()-t0:.0f}s)"
    )


def main():
    ap = argparse.ArgumentParser(description="Chunk corpus per ingestion_spec (no embeddings)")
    ap.add_argument("--parquet", default=None, help="Path to documents test.parquet")
    ap.add_argument("--out", default="indexes/chunk_v1", help="Output directory")
    ap.add_argument("--token-mode", choices=["approx", "bge_m3"], default="bge_m3")
    ap.add_argument("--limit", type=int, default=None, help="Optional doc cap for smoke tests")
    ap.add_argument("--build-id", default="chunk_v1")
    ap.add_argument("--write-jsonl", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--load-postgres", action="store_true", help="Load after chunking")
    ap.add_argument(
        "--load-only",
        action="store_true",
        help="Skip chunking; only load existing --out into Postgres",
    )
    ap.add_argument(
        "--dsn",
        default=os.environ.get(
            "ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"
        ),
    )
    args = ap.parse_args()

    out = Path(args.out)
    if args.load_only:
        load_postgres(out, args.build_id, args.dsn)
        return

    parquet = find_parquet(args.parquet)
    run_chunk(
        parquet=parquet,
        out_dir=out,
        token_mode=args.token_mode,
        limit=args.limit,
        write_jsonl=args.write_jsonl,
        build_id=args.build_id,
    )
    if args.load_postgres:
        load_postgres(out, args.build_id, args.dsn)


if __name__ == "__main__":
    main()
