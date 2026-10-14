"""Build SQLite FTS5 BM25 index from PostgreSQL chunks table.

Creates a dedicated full-text search index for high-speed BM25 sparse retrieval.
Uses Porter stemming and unicode tokenization.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_index(
    dsn: str,
    sqlite_path: Path,
    batch_size: int = 50000,
    force: bool = False,
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    if sqlite_path.exists() and not force:
        conn_sq = sqlite3.connect(sqlite_path)
        cur_sq = conn_sq.cursor()
        try:
            cur_sq.execute("SELECT count(*) FROM chunks_fts;")
            count = cur_sq.fetchone()[0]
            if count >= 540000:
                print(f"[BM25] Existing FTS5 index found at {sqlite_path} with {count:,} chunks. Skipping build.")
                conn_sq.close()
                return
        except Exception:
            pass
        conn_sq.close()

    if sqlite_path.exists():
        sqlite_path.unlink()

    print(f"[BM25] Building SQLite FTS5 index at {sqlite_path} from Postgres...")
    conn_sq = sqlite3.connect(sqlite_path)
    cur_sq = conn_sq.cursor()

    cur_sq.execute("PRAGMA journal_mode = OFF;")
    cur_sq.execute("PRAGMA synchronous = OFF;")
    cur_sq.execute("PRAGMA cache_size = -1000000;")  # 1GB cache

    cur_sq.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            doc_id,
            content,
            tokenize = 'porter unicode61'
        );
    """)

    conn_pg = psycopg.connect(dsn)
    cur_pg = conn_pg.cursor(name="chunk_stream")

    cur_pg.execute("""
        SELECT chunk_id, doc_id, coalesce(header, '') || ' ' || text
        FROM chunks
        ORDER BY chunk_id;
    """)

    total_inserted = 0
    t0 = time.time()

    while True:
        rows = cur_pg.fetchmany(batch_size)
        if not rows:
            break
        cur_sq.executemany("INSERT INTO chunks_fts(chunk_id, doc_id, content) VALUES (?, ?, ?);", rows)
        conn_sq.commit()
        total_inserted += len(rows)
        elapsed = time.time() - t0
        rate = total_inserted / max(elapsed, 0.001)
        print(f"[BM25] Inserted {total_inserted:,} chunks ({rate:,.0f} chunks/sec)...")

    cur_pg.close()
    conn_pg.close()

    print(f"[BM25] Optimizing FTS5 index...")
    cur_sq.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize');")
    conn_sq.commit()
    conn_sq.close()

    total_time = time.time() - t0
    file_size_mb = sqlite_path.stat().st_size / (1024 * 1024)
    print(f"[BM25] Complete! Indexed {total_inserted:,} chunks in {total_time:.1f}s ({file_size_mb:.1f} MB on disk).")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 FTS5 index for Enterprise RAG Bench")
    parser.add_argument(
        "--sqlite-path",
        type=str,
        default="indexes/bm25_fts5.db",
        help="Target SQLite DB path for FTS5 index",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=os.environ.get("ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"),
    )
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_index(
        dsn=args.dsn,
        sqlite_path=Path(args.sqlite_path),
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
