"""Dense retriever with HNSW vector similarity search and chunk-to-doc collapse.

Implements Docs/ingestion_spec.md §6.5:
  "Hits are chunks → emit unique doc_id, best rank wins (highest score / earliest rank).
   Tie-break: lower chunk_index."

Queries Postgres with pgvector using cosine distance (<=>).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import psycopg
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.encoder import QueryEncoder

DEFAULT_DSN = os.environ.get(
    "ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"
)
DEFAULT_BUILD_ID = "chunk_v1"


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_type: str
    chunk_index: int
    score: float  # cosine similarity in [-1, 1]
    distance: float  # cosine distance in [0, 2]
    header: str
    text: str


@dataclass
class RetrievedDocument:
    doc_id: str
    rank: int  # 1-indexed
    score: float  # score of best chunk
    source_type: str
    best_chunk_id: str
    best_chunk_index: int
    best_chunk_header: str
    best_chunk_text: str
    chunks: List[RetrievedChunk] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    documents: List[RetrievedDocument]
    raw_chunks: List[RetrievedChunk]
    elapsed_sec: float

    @property
    def document_ids(self) -> List[str]:
        """Rank-ordered list of unique document IDs."""
        return [d.doc_id for d in self.documents]


class DenseRetriever:
    """HNSW Dense Retriever over Postgres chunks with chunk-to-document collapse."""

    def __init__(
        self,
        dsn: str = DEFAULT_DSN,
        build_id: str = DEFAULT_BUILD_ID,
        encoder: Optional[QueryEncoder] = None,
        ef_search: int = 64,
    ):
        self.dsn = dsn or DEFAULT_DSN
        self.build_id = build_id
        self.ef_search = ef_search
        self._encoder = encoder
        self._conn: Optional[psycopg.Connection] = None

    @property
    def encoder(self) -> QueryEncoder:
        if self._encoder is None:
            self._encoder = QueryEncoder()
        return self._encoder

    def _get_connection(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                self.dsn,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
                autocommit=True,
            )
            register_vector(self._conn)
            with self._conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {int(self.ef_search)};")
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def collapse_chunks(
        self, chunks: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedDocument]:
        """Collapse multiple chunk hits from the same document to a single unique doc_id.

        Best chunk rank determines the document rank. All chunks for the selected
        document from the candidate pool are retained.
        """
        seen_docs: Dict[str, RetrievedDocument] = {}
        ordered_docs: List[RetrievedDocument] = []

        for ch in chunks:
            if ch.doc_id not in seen_docs:
                if len(ordered_docs) >= top_k:
                    continue
                doc = RetrievedDocument(
                    doc_id=ch.doc_id,
                    rank=len(ordered_docs) + 1,
                    score=ch.score,
                    source_type=ch.source_type,
                    best_chunk_id=ch.chunk_id,
                    best_chunk_index=ch.chunk_index,
                    best_chunk_header=ch.header,
                    best_chunk_text=ch.text,
                    chunks=[ch],
                )
                seen_docs[ch.doc_id] = doc
                ordered_docs.append(doc)
            else:
                # Document already seen; append secondary chunk if within candidate set
                seen_docs[ch.doc_id].chunks.append(ch)

        return ordered_docs

    def retrieve_by_vector(
        self,
        query_vector: Sequence[float],
        query_text: str = "",
        top_k: int = 20,
        fetch_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """Search nearest chunks given a precomputed query vector, then collapse to top-K documents."""
        t0 = time.time()
        # Fetch extra chunks to guarantee top_k unique documents after collapse
        fetch_k = fetch_k or max(top_k * 4, 100)

        conn = self._get_connection()
        query_vec_list = list(query_vector)

        if source_filter:
            sql = """
                SELECT chunk_id, doc_id, source_type, chunk_index, header, text,
                       (embedding <=> %s::vector) AS distance,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE build_id = %s AND source_type = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params = (query_vec_list, query_vec_list, self.build_id, source_filter, query_vec_list, fetch_k)
        else:
            sql = """
                SELECT chunk_id, doc_id, source_type, chunk_index, header, text,
                       (embedding <=> %s::vector) AS distance,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE build_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params = (query_vec_list, query_vec_list, self.build_id, query_vec_list, fetch_k)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        raw_chunks: List[RetrievedChunk] = []
        for r in rows:
            raw_chunks.append(
                RetrievedChunk(
                    chunk_id=r[0],
                    doc_id=r[1],
                    source_type=r[2],
                    chunk_index=r[3],
                    header=r[4],
                    text=r[5],
                    distance=float(r[6]),
                    score=float(r[7]),
                )
            )

        documents = self.collapse_chunks(raw_chunks, top_k=top_k)
        elapsed = time.time() - t0

        return RetrievalResult(
            query=query_text,
            documents=documents,
            raw_chunks=raw_chunks,
            elapsed_sec=elapsed,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        fetch_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """Encode query string and retrieve collapsed top-K documents."""
        q_emb = self.encoder.encode([query])[0]
        return self.retrieve_by_vector(
            query_vector=q_emb,
            query_text=query,
            top_k=top_k,
            fetch_k=fetch_k,
            source_filter=source_filter,
        )

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int = 20,
        fetch_k: Optional[int] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[RetrievalResult]:
        """Batch encode queries and retrieve top-K collapsed documents for each."""
        print(f"[DenseRetriever] Encoding {len(queries)} queries...")
        t0 = time.time()
        vectors = self.encoder.encode(queries, batch_size=batch_size, show_progress=show_progress)
        print(f"[DenseRetriever] Query encoding completed in {time.time() - t0:.2f}s.")

        print(f"[DenseRetriever] Running HNSW retrieval for {len(queries)} queries...")
        results: List[RetrievalResult] = []
        t1 = time.time()

        for idx, (q_text, q_vec) in enumerate(zip(queries, vectors)):
            res = self.retrieve_by_vector(
                query_vector=q_vec,
                query_text=q_text,
                top_k=top_k,
                fetch_k=fetch_k,
            )
            results.append(res)
            if show_progress and (idx + 1) % 50 == 0:
                print(f"  retrieved {idx + 1}/{len(queries)} questions ({time.time() - t1:.1f}s)...")

        print(f"[DenseRetriever] Batch retrieval finished in {time.time() - t1:.2f}s.")
        return results


def smoke_test(retriever: DenseRetriever):
    """Run smoke test on catalog qst_0001 and verify gold document recovery."""
    catalog_path = ROOT / "Data" / "benchmark_catalog.jsonl"
    if not catalog_path.exists():
        print(f"Error: {catalog_path} not found.")
        return

    with open(catalog_path, encoding="utf-8") as f:
        first_q = json.loads(f.readline())

    qid = first_q["question_id"]
    query = first_q["question"]
    expected_ids = first_q.get("expected_doc_ids", [])
    expected_sources = first_q.get("source_types", [])

    print("\n" + "=" * 80)
    print(f"SMOKE TEST: {qid}")
    print(f"Query: {query}")
    print(f"Expected Gold Docs: {expected_ids}")
    print(f"Expected Gold Sources: {expected_sources}")
    print("=" * 80)

    res = retriever.retrieve(query, top_k=20)
    retrieved_ids = res.document_ids

    print(f"\nRetrieval completed in {res.elapsed_sec*1000:.1f} ms")
    print(f"Unique documents returned: {len(retrieved_ids)}")
    print("\nTop 5 Retrieved Documents:")
    for doc in res.documents[:5]:
        is_gold = "⭐ GOLD HIT" if doc.doc_id in expected_ids else ""
        print(f"  Rank {doc.rank:2d}: [{doc.source_type}] {doc.doc_id} (score={doc.score:.4f}, chunks={len(doc.chunks)}) {is_gold}")
        print(f"           Header: {doc.best_chunk_header}")

    # Check rank of expected doc
    for gold_id in expected_ids:
        if gold_id in retrieved_ids:
            rank = retrieved_ids.index(gold_id) + 1
            print(f"\nSUCCESS: Gold doc {gold_id} found at Rank {rank}!")
        else:
            print(f"\nGold doc {gold_id} not in top-20.")


def main():
    parser = argparse.ArgumentParser(description="Dense retriever over EnterpriseRAGBench chunks")
    parser.add_argument("--query", type=str, default=None, help="Query string to search")
    parser.add_argument("--top-k", type=int, default=10, help="Number of collapsed documents to return")
    parser.add_argument("--ef-search", type=int, default=64, help="HNSW ef_search parameter")
    parser.add_argument("--build-id", type=str, default=DEFAULT_BUILD_ID)
    parser.add_argument("--dsn", type=str, default=DEFAULT_DSN)
    parser.add_argument("--show-text", action="store_true", help="Print chunk text snippet")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test on catalog qst_0001")
    args = parser.parse_args()

    retriever = DenseRetriever(
        dsn=args.dsn,
        build_id=args.build_id,
        ef_search=args.ef_search,
    )

    if args.smoke_test or args.query is None:
        smoke_test(retriever)
        return

    res = retriever.retrieve(args.query, top_k=args.top_k)
    print(f"\nQuery: {args.query}")
    print(f"Elapsed: {res.elapsed_sec*1000:.1f} ms | Documents returned: {len(res.documents)}")
    print("-" * 80)
    for doc in res.documents:
        print(f"Rank {doc.rank:2d}: [{doc.source_type}] {doc.doc_id} | score: {doc.score:.4f}")
        print(f"  Header: {doc.best_chunk_header}")
        if args.show_text:
            snippet = doc.best_chunk_text.replace("\n", " ")[:200]
            print(f"  Snippet: {snippet}...")
    print("-" * 80)


if __name__ == "__main__":
    main()
