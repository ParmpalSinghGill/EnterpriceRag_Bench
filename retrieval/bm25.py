"""BM25 Sparse Retrieval using SQLite FTS5 index.

Performs fast full-text keyword matching with chunk-to-doc collapse.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "indexes" / "bm25_fts5.db"

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your"
}


def clean_fts5_query(query: str) -> str:
    """Sanitize raw query string into valid FTS5 match syntax with stop words filtered."""
    tokens = re.findall(r"\b[A-Za-z0-9_.-]+\b", query)
    if not tokens:
        return '""'
    cleaned = []
    for t in tokens:
        t_lower = t.lower()
        if t_lower not in STOP_WORDS and len(t_lower) >= 2:
            cleaned.append(f'"{t.replace(chr(34), chr(34)+chr(34))}"')
    if not cleaned:
        # Fallback to all non-trivial tokens if all tokens were stop words
        cleaned = [f'"{t.replace(chr(34), chr(34)+chr(34))}"' for t in tokens if len(t) >= 2]
    if not cleaned:
        return '""'
    return " OR ".join(cleaned)


class BM25Retriever:
    """Sparse BM25 retriever backed by SQLite FTS5."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"BM25 FTS5 index not found at {self.db_path}. Run build_bm25_index first.")
        # URI mode ro (read-only) allows safe concurrent access across threads
        self.conn = sqlite3.connect(f"file:{self.db_path.resolve()}?mode=ro", uri=True, check_same_thread=False)

    def search_raw(self, query: str, top_k: int = 100) -> List[Tuple[str, str, float]]:
        """Search chunks and return (doc_id, chunk_id, bm25_score).

        Note: SQLite bm25() returns negative numbers where more negative = better match.
        We negate it so higher score = better match.
        """
        match_query = clean_fts5_query(query)
        if match_query == '""':
            return []

        cur = self.conn.cursor()
        sql = """
            SELECT doc_id, chunk_id, -bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?;
        """
        try:
            cur.execute(sql, (match_query, top_k))
            return cur.fetchall()
        except sqlite3.OperationalError:
            return []

    def retrieve_docs(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Retrieve documents collapsed by doc_id (taking max score per doc)."""
        raw_hits = self.search_raw(query, top_k=top_k * 4)
        best_doc_scores: Dict[str, float] = {}

        for doc_id, chunk_id, score in raw_hits:
            if doc_id not in best_doc_scores or score > best_doc_scores[doc_id]:
                best_doc_scores[doc_id] = score

        ranked = sorted(best_doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def close(self) -> None:
        if self.conn:
            self.conn.close()
