# Phase-1 chunking (text only)

Implements [`Docs/ingestion_spec.md`](../Docs/ingestion_spec.md) `sentence_quarter_v1`.

```bash
# smoke
python -m ingestion.run_chunk --limit 1000 --out indexes/chunk_smoke

# full size report
python -m ingestion.run_chunk --out indexes/chunk_v1

# optional: load into Docker Postgres (see Docs/chunk_store.md)
python -m ingestion.run_chunk --out indexes/chunk_v1 --load-postgres
```

Embeddings are intentionally deferred.
