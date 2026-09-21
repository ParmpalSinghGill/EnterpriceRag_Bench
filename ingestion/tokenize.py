"""Token counting for sentence_quarter_v1.

Modes:
  approx  — chars/4 planner (fast; for size dry-runs only)
  bge_m3  — BAAI/bge-m3 tokenizer.json (spec-accurate packing / hard-split)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, Optional

TokenMode = Literal["approx", "bge_m3"]

_LOCAL_BGE_M3_TOKENIZER = Path(
    "/home/parmpal/.cache/huggingface/hub/models--BAAI--bge-m3/"
    "snapshots/5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json"
)


def _resolve_bge_tokenizer_path() -> str:
    env = os.environ.get("ERAG_BGE_M3_TOKENIZER")
    if env and Path(env).exists():
        return env
    if _LOCAL_BGE_M3_TOKENIZER.exists():
        return str(_LOCAL_BGE_M3_TOKENIZER)
    from huggingface_hub import hf_hub_download

    return hf_hub_download("BAAI/bge-m3", "tokenizer.json")


class TokenCounter:
    def __init__(self, mode: TokenMode = "approx", cache_size: int = 100_000):
        self.mode = mode
        self._tok = None
        self._cache: Dict[int, int] = {}
        self._cache_size = cache_size
        if mode == "bge_m3":
            from tokenizers import Tokenizer

            self._tok = Tokenizer.from_file(_resolve_bge_tokenizer_path())
            self._tok.no_padding()
            self._tok.no_truncation()

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.mode == "approx":
            return max(1, len(text) // 4)
        key = hash(text)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        n = len(self._tok.encode(text).ids)
        if len(self._cache) >= self._cache_size:
            # drop an arbitrary half to bound memory
            for i, k in enumerate(list(self._cache.keys())):
                if i >= self._cache_size // 2:
                    break
                del self._cache[k]
        self._cache[key] = n
        return n


@lru_cache(maxsize=2)
def get_counter(mode: TokenMode = "approx") -> TokenCounter:
    return TokenCounter(mode)
