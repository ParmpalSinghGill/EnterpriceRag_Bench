"""Hybrid Retrieval combining Dense (BGE-M3) and Sparse (BM25 FTS5).

Implements Reciprocal Rank Fusion (RRF):
    RRF(doc) = w_dense / (k + rank_dense(doc)) + w_sparse / (k + rank_sparse(doc))

Standard RRF constant k=20, weight_sparse=1.2.
Collapses chunks to unique documents before or during fusion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever, RetrievedDocument


@dataclass
class HybridDocument:
    doc_id: str
    rank: int
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


class HybridRetriever:
    """Combines BGE-M3 dense similarity search with BM25 sparse keyword search."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        dsn: Optional[str] = None,
        bm25_db_path: Optional[Path] = None,
        rrf_k: int = 20,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.2,
    ):
        self.dense = dense_retriever or (DenseRetriever(dsn=dsn) if dsn else DenseRetriever())
        self.sparse = bm25_retriever or BM25Retriever(
            db_path=bm25_db_path or (ROOT / "indexes" / "bm25_fts5.db")
        )
        self.rrf_k = rrf_k
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse

    def fuse_rankings(
        self,
        dense_docs: Sequence[RetrievedDocument],
        sparse_docs: Sequence[Tuple[str, float]],
        top_k: int = 20,
    ) -> List[HybridDocument]:
        """Combine dense and sparse candidate lists using Reciprocal Rank Fusion."""
        dense_ranks: Dict[str, Tuple[int, float]] = {
            d.doc_id: (idx + 1, d.score) for idx, d in enumerate(dense_docs)
        }
        sparse_ranks: Dict[str, Tuple[int, float]] = {
            doc_id: (idx + 1, score) for idx, (doc_id, score) in enumerate(sparse_docs)
        }

        all_doc_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        rrf_scores: Dict[str, float] = {}

        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in dense_ranks:
                r_dense, _ = dense_ranks[doc_id]
                score += self.weight_dense / (self.rrf_k + r_dense)
            if doc_id in sparse_ranks:
                r_sparse, _ = sparse_ranks[doc_id]
                score += self.weight_sparse / (self.rrf_k + r_sparse)
            rrf_scores[doc_id] = score

        # Sort descending by RRF score; tie-break: earlier dense rank, then doc_id
        def sort_key(d_id: str) -> Tuple[float, int, str]:
            score = rrf_scores[d_id]
            dense_r = dense_ranks[d_id][0] if d_id in dense_ranks else 999999
            return (score, -dense_r, d_id)

        sorted_doc_ids = sorted(all_doc_ids, key=sort_key, reverse=True)

        results: List[HybridDocument] = []
        for rank, doc_id in enumerate(sorted_doc_ids[:top_k], start=1):
            d_info = dense_ranks.get(doc_id)
            s_info = sparse_ranks.get(doc_id)
            results.append(
                HybridDocument(
                    doc_id=doc_id,
                    rank=rank,
                    score=rrf_scores[doc_id],
                    dense_rank=d_info[0] if d_info else None,
                    dense_score=d_info[1] if d_info else None,
                    sparse_rank=s_info[0] if s_info else None,
                    sparse_score=s_info[1] if s_info else None,
                )
            )

        return results

    def retrieve_docs(
        self,
        query: str,
        top_k: int = 20,
        candidate_pool: int = 100,
    ) -> List[HybridDocument]:
        """Retrieve top_k documents by fusing dense and sparse search rankings."""
        dense_res = self.dense.retrieve(
            query=query,
            top_k=candidate_pool,
        )
        sparse_docs = self.sparse.retrieve_docs(
            query=query,
            top_k=candidate_pool,
        )
        return self.fuse_rankings(
            dense_docs=dense_res.documents,
            sparse_docs=sparse_docs,
            top_k=top_k,
        )

    def batch_retrieve_docs(
        self,
        queries: Sequence[str],
        top_k: int = 20,
        candidate_pool: int = 100,
        batch_size: int = 32,
    ) -> List[List[HybridDocument]]:
        """Batch retrieve top_k documents for a sequence of queries."""
        dense_results = self.dense.batch_retrieve(
            queries=queries,
            top_k=candidate_pool,
            batch_size=batch_size,
        )
        all_hybrid: List[List[HybridDocument]] = []
        for query, dense_res in zip(queries, dense_results):
            sparse_docs = self.sparse.retrieve_docs(
                query=query,
                top_k=candidate_pool,
            )
            hybrid_docs = self.fuse_rankings(
                dense_docs=dense_res.documents,
                sparse_docs=sparse_docs,
                top_k=top_k,
            )
            all_hybrid.append(hybrid_docs)
        return all_hybrid

    def fuse_multi_rankings(
        self,
        ranked_lists: Sequence[Tuple[Sequence[str], float]],
        top_k: int = 20,
    ) -> List[HybridDocument]:
        """Combine arbitrary multiple candidate rankings using Reciprocal Rank Fusion."""
        all_doc_ids = set()
        ranks_by_mode: List[Dict[str, int]] = []
        weights: List[float] = []

        for doc_list, weight in ranked_lists:
            r_map = {doc_id: idx + 1 for idx, doc_id in enumerate(doc_list)}
            ranks_by_mode.append(r_map)
            weights.append(weight)
            all_doc_ids.update(r_map.keys())

        rrf_scores: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            score = 0.0
            for r_map, w in zip(ranks_by_mode, weights):
                if doc_id in r_map:
                    score += w / (self.rrf_k + r_map[doc_id])
            rrf_scores[doc_id] = score

        sorted_doc_ids = sorted(all_doc_ids, key=lambda d: rrf_scores[d], reverse=True)

        results: List[HybridDocument] = []
        for rank, doc_id in enumerate(sorted_doc_ids[:top_k], start=1):
            results.append(
                HybridDocument(
                    doc_id=doc_id,
                    rank=rank,
                    score=rrf_scores[doc_id],
                )
            )
        return results

    def retrieve_with_expansion(
        self,
        query: str,
        expanded_keywords: str,
        hyde_passage: str,
        top_k: int = 20,
        candidate_pool: int = 100,
    ) -> List[HybridDocument]:
        """Retrieve using Original Dense, Original BM25, HyDE Dense, and Expanded BM25."""
        d_orig = self.dense.retrieve(query=query, top_k=candidate_pool)
        b_orig = self.sparse.retrieve_docs(query=query, top_k=candidate_pool)
        d_hyde = self.dense.retrieve(query=hyde_passage, top_k=candidate_pool)
        b_exp = self.sparse.retrieve_docs(query=expanded_keywords, top_k=candidate_pool)

        ranked_lists = [
            ([d.doc_id for d in d_orig.documents], self.weight_dense * 0.8),
            ([d for d, _ in b_orig], self.weight_sparse * 0.85),
            ([d.doc_id for d in d_hyde.documents], self.weight_dense * 1.2),
            ([d for d, _ in b_exp], self.weight_sparse * 1.5),
        ]
        return self.fuse_multi_rankings(ranked_lists, top_k=top_k)

    def batch_retrieve_with_expansion(
        self,
        items: Sequence[Tuple[str, str, str]],  # (query, expanded_keywords, hyde_passage)
        top_k: int = 20,
        candidate_pool: int = 100,
        batch_size: int = 32,
    ) -> List[List[HybridDocument]]:
        """Batch retrieve top-K documents using multi-modal query expansion."""
        orig_queries = [it[0] for it in items]
        exp_keywords = [it[1] for it in items]
        hyde_passages = [it[2] for it in items]

        # 1. Batch dense retrieval for orig + hyde
        all_dense_queries = orig_queries + hyde_passages
        all_dense_res = self.dense.batch_retrieve(
            queries=all_dense_queries,
            top_k=candidate_pool,
            batch_size=batch_size,
        )
        n = len(items)
        orig_dense_res = all_dense_res[:n]
        hyde_dense_res = all_dense_res[n:]

        all_results: List[List[HybridDocument]] = []
        for i in range(n):
            q_orig = orig_queries[i]
            q_exp = exp_keywords[i]

            b_orig = self.sparse.retrieve_docs(query=q_orig, top_k=candidate_pool)
            b_exp = self.sparse.retrieve_docs(query=q_exp, top_k=candidate_pool)
            d_orig = orig_dense_res[i]
            d_hyde = hyde_dense_res[i]

            ranked_lists = [
                ([d.doc_id for d in d_orig.documents], self.weight_dense * 0.8),
                ([d for d, _ in b_orig], self.weight_sparse * 0.85),
                ([d.doc_id for d in d_hyde.documents], self.weight_dense * 1.2),
                ([d for d, _ in b_exp], self.weight_sparse * 1.5),
            ]
            fused = self.fuse_multi_rankings(ranked_lists, top_k=top_k)
            all_results.append(fused)

        return all_results

    def close(self) -> None:
        if self.dense:
            self.dense.close()
        if self.sparse:
            self.sparse.close() 