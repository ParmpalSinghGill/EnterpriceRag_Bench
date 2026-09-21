# Chunk store setup — Docker Postgres (pgvector) + phase-1 chunking

This guide covers **phase 1 only**: split the corpus with [`ingestion_spec.md`](ingestion_spec.md) and measure size. **No embeddings yet.** The `embedding vector(1024)` column is created but left `NULL`.

Contracts:

- Chunking rules: [`ingestion_spec.md`](ingestion_spec.md) (`sentence_quarter_v1`, `C=8192`, `T=2048`)
- Scoring later: [`evaluation_spec.md`](evaluation_spec.md)

---

## 1. What you get

| Piece | Path | Role |
|---|---|---|
| Docker Compose | `docker-compose.yml` | Postgres 16 + pgvector on port **5433** |
| SQL init | `infra/postgres/init.sql` | `documents`, `chunks`, `ingest_builds` |
| Chunker | `ingestion/chunker.py` | Spec §5–§7 splitter |
| CLI | `ingestion/run_chunk.py` | Dry-run size report + optional DB load |

Default DB URL:

```text
postgresql://erag:erag@127.0.0.1:5433/erag
```

---

## 2. Start Postgres (Docker)

Prerequisites: Docker Engine + Compose plugin.

From the repo root:

```bash
docker compose up -d
docker compose ps
docker compose logs -f erag-postgres   # optional
```

Wait until healthy (`pg_isready`). Init SQL runs only on **first** volume create. To reset the volume:

```bash
docker compose down -v
docker compose up -d
```

Stop (keep data):

```bash
docker compose down
```

---

## 3. Python deps for chunking

```bash
conda activate ERAG   # or your env
pip install pyarrow psycopg[binary] transformers
# datasets optional if you load via HF instead of local parquet
```

The corpus parquet is expected from the Hugging Face cache (already used by this project):

```text
.../datasets--onyx-dot-app--EnterpriseRAG-Bench/snapshots/<rev>/data/documents/test.parquet
```

Or set:

```bash
export ERAG_DOCUMENTS_PARQUET=/absolute/path/to/test.parquet
```

---

## 4. Run chunking (size check — do this first)

### 4.1 Smoke test (1k docs)

```bash
cd /path/to/EnterpriseRAGBench
python -m ingestion.run_chunk \
  --limit 1000 \
  --token-mode approx \
  --out indexes/chunk_smoke \
  --build-id chunk_smoke
```

### 4.2 Full corpus (≈512k docs)

```bash
python -m ingestion.run_chunk \
  --token-mode approx \
  --out indexes/chunk_v1 \
  --build-id chunk_v1
```

`approx` = `chars/4` token planner (fast). For packing that matches the future `bge-m3` embedder:

```bash
python -m ingestion.run_chunk --token-mode bge_m3 --out indexes/chunk_v1_bge --build-id chunk_v1_bge
```

(Slower; downloads tokenizer on first run.)

### 4.3 Load existing chunks only (no re-split)

If `indexes/chunk_v1/chunks.jsonl` already exists:

```bash
docker compose up -d
python -m ingestion.run_chunk --load-only --out indexes/chunk_v1 --build-id chunk_v1
```

### 4.4 Outputs

```text
indexes/<build>/
  manifest.json       # pack, C/T/O, counts
  size_report.json    # docs/chunks, token stats, on-disk bytes
  docid_map.jsonl     # chunk_id → doc_id metadata
  chunks.jsonl        # full chunk text (large)
```

Inspect:

```bash
python - <<'PY'
import json
r=json.load(open("indexes/chunk_v1/size_report.json"))
print("docs", r["n_docs"], "chunks", r["n_chunks"], "mean", r["chunks_per_doc_mean"])
print("token_stats", r["token_stats"])
print("disk", r["on_disk_human"])
print("by_source sample:", list(r["by_source"].items())[:3])
PY
```

What to look for:

- `token_stats.over_C` should be **0** (blocking rule: no chunk > 8192)
- `chunks_per_doc_mean` roughly **1.0–1.3** with `approx` @ T=2048
- Multi-chunk % high mainly for `confluence` / `fireflies`

---

## 5. Load chunks into Postgres (still no embeddings)

After a successful chunk run **with** `chunks.jsonl`:

```bash
docker compose up -d   # if not already running

python -m ingestion.run_chunk \
  --token-mode approx \
  --out indexes/chunk_v1 \
  --build-id chunk_v1 \
  --load-postgres
```

Or only load an existing out dir (re-runs chunk then load). To load without re-chunking, use Python:

```bash
python - <<'PY'
from pathlib import Path
from ingestion.run_chunk import load_postgres
load_postgres(Path("indexes/chunk_v1"), "chunk_v1", "postgresql://erag:erag@127.0.0.1:5433/erag")
PY
```

Verify:

```bash
docker exec -it erag-postgres psql -U erag -d erag -c "\
  SELECT build_id, n_docs, n_chunks FROM ingest_builds; \
  SELECT source_type, COUNT(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC; \
  SELECT COUNT(*) FILTER (WHERE embedding IS NULL) AS null_embeddings, COUNT(*) AS n FROM chunks;"
```

---

## 6. Schema notes (phase 1)

- `chunks.embedding vector(1024)` — **reserved** for `BAAI/bge-m3`; leave `NULL`
- No HNSW / IVFFlat index yet (useless without vectors)
- Sparse/BM25 is **future work** (see ingestion_spec §10)
- Port **5433** avoids clashing with a local Postgres on 5432

---

## 7. Suggested order

1. `docker compose up -d`
2. Smoke chunk `--limit 1000` → read `size_report.json`
3. Full chunk → confirm size / `over_C == 0`
4. `--load-postgres`
5. **Later:** embedding pass filling `chunks.embedding`

---

## 8. Troubleshooting

| Issue | Fix |
|---|---|
| Parquet not found | Pass `--parquet` or `ERAG_DOCUMENTS_PARQUET` |
| Port 5433 in use | Change host port in `docker-compose.yml` |
| Init SQL didn’t apply | `docker compose down -v` then `up -d` (wipes DB) |
| `psycopg` missing | `pip install 'psycopg[binary]'` |
| Disk full on `chunks.jsonl` | Use `--no-write-jsonl` for map+stats only (cannot load DB without jsonl) |
