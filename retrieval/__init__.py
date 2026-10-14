"""Retrieval module for EnterpriseRAGBench.

Includes dense vector retrieval, query encoding, and chunk-to-document collapse.
"""

from typing import Any

__all__ = [
    "QueryEncoder",
    "DenseRetriever",
    "RetrievedChunk",
    "RetrievedDocument",
    "RetrievalResult",
]


def __getattr__(name: str) -> Any:
    if name == "QueryEncoder":
        from retrieval.encoder import QueryEncoder
        return QueryEncoder
    if name in ("DenseRetriever", "RetrievedChunk", "RetrievedDocument", "RetrievalResult"):
        import retrieval.dense as d
        return getattr(d, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
