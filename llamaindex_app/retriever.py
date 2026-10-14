"""LlamaIndex Custom Retriever for Enterprise Hybrid Search.

Integrates:
- Dense BGE-M3 vector search on PostgreSQL pgvector (HNSW)
- Sparse BM25 lexical search on SQLite FTS5
- Enterprise domain grounding & HyDE query expansion
- Top-K chunk-to-node conversion for LlamaIndex
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

ROOT = Path(__file__).resolve().parents[1]

from retrieval.hybrid import HybridRetriever, HybridDocument
from retrieval.query_expansion import QueryExpander

DEFAULT_DSN = os.environ.get(
    "ERAG_DATABASE_URL", "postgresql://erag:erag@127.0.0.1:5433/erag"
)

_SHARED_HYBRID: Optional[HybridRetriever] = None
_SHARED_EXPANDER: Optional[QueryExpander] = None


def get_shared_hybrid_retriever(
    dsn: str = DEFAULT_DSN,
    bm25_db_path: Optional[Path] = None,
    rrf_k: int = 20,
    weight_dense: float = 1.0,
    weight_sparse: float = 1.2,
) -> HybridRetriever:
    global _SHARED_HYBRID
    if _SHARED_HYBRID is None:
        _SHARED_HYBRID = HybridRetriever(
            dsn=dsn,
            bm25_db_path=bm25_db_path,
            rrf_k=rrf_k,
            weight_dense=weight_dense,
            weight_sparse=weight_sparse,
        )
    return _SHARED_HYBRID


def get_shared_query_expander() -> QueryExpander:
    global _SHARED_EXPANDER
    if _SHARED_EXPANDER is None:
        _SHARED_EXPANDER = QueryExpander()
    return _SHARED_EXPANDER


class EnterpriseHybridRetriever(BaseRetriever):
    """LlamaIndex-compatible hybrid retriever fusing Dense BGE-M3 and Sparse BM25."""

    def __init__(
        self,
        top_k: int = 7,
        use_expansion: bool = True,
        max_chunks_per_doc: int = 2,
        dsn: str = DEFAULT_DSN,
        bm25_db_path: Optional[Path] = None,
        rrf_k: int = 20,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.2,
    ):
        super().__init__()
        self.top_k = top_k
        self.use_expansion = use_expansion
        self.max_chunks_per_doc = max_chunks_per_doc
        self.dsn = dsn
        self.hybrid = get_shared_hybrid_retriever(
            dsn=dsn,
            bm25_db_path=bm25_db_path,
            rrf_k=rrf_k,
            weight_dense=weight_dense,
            weight_sparse=weight_sparse,
        )
        self.expander = get_shared_query_expander() if use_expansion else None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Execute multi-stream retrieval and convert results to LlamaIndex Nodes."""
        query = query_bundle.query_str

        # 1. Hybrid retrieval (with optional domain-grounded HyDE)
        if self.use_expansion and self.expander:
            exp = self.expander.expand(query)
            h_docs: List[HybridDocument] = self.hybrid.retrieve_with_expansion(
                query=query,
                expanded_keywords=exp.get("expanded_keywords", query),
                hyde_passage=exp.get("hyde_passage", query),
                top_k=self.top_k,
            )
        else:
            h_docs = self.hybrid.retrieve_docs(query, top_k=self.top_k)

        # 2. Pre-fetch context chunks from Postgres
        doc_ids = [d.doc_id for d in h_docs]
        doc_score_map = {d.doc_id: d.score for d in h_docs}

        nodes: List[NodeWithScore] = []
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                for doc_id in doc_ids:
                    cur.execute(
                        """
                        SELECT chunk_id, source_type, chunk_index, header, text
                        FROM chunks
                        WHERE doc_id = %s
                        ORDER BY chunk_index
                        LIMIT %s;
                        """,
                        (doc_id, self.max_chunks_per_doc),
                    )
                    for r in cur.fetchall():
                        chunk_id, source_type, chunk_idx, header, text = r
                        header_str = header or f"[{source_type}] Document"
                        formatted_text = f"--- {header_str} ---\n{text.strip()}"

                        node = TextNode(
                            text=formatted_text,
                            id_=chunk_id,
                            metadata={
                                "doc_id": doc_id,
                                "source_type": source_type,
                                "chunk_index": chunk_idx,
                                "header": header_str,
                            },
                        )
                        nodes.append(
                            NodeWithScore(
                                node=node,
                                score=doc_score_map.get(doc_id, 0.0),
                            )
                        )

        return nodes

