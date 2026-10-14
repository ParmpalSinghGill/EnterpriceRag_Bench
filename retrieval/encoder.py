"""Query encoder for BAAI/bge-m3 dense retrieval.

Encodes natural language questions into 1024-dimensional normalized vectors,
matching the exact pooling and normalization used during corpus chunk ingestion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "BAAI/bge-m3"
DIM = 1024
C_MAX = 8192

_LOCAL_SNAPS = [
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/9a0624b896d81da7492a910ffa53731274b6cf3d",
]


def resolve_model_path(explicit: Optional[str] = None) -> str:
    """Find local snapshot if present, else fallback to Hugging Face ID."""
    if explicit and Path(explicit).exists():
        return explicit
    env = os.environ.get("ERAG_BGE_M3_PATH")
    if env and Path(env).exists():
        return env
    for p in _LOCAL_SNAPS:
        if (p / "config.json").exists() and (
            (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists()
        ):
            return str(p)
    return MODEL_ID


class QueryEncoder:
    """Encodes queries using BAAI/bge-m3."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        fp16: bool = True,
        max_length: int = C_MAX,
    ):
        self.model_path = resolve_model_path(model_path)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.fp16 = fp16 and self.device.startswith("cuda")
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.max_length = max_length

        print(f"[QueryEncoder] Loading {self.model_path} onto {self.device} (dtype={self.dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.eval()
        self.model.to(device=self.device, dtype=self.dtype)
        print("[QueryEncoder] Model loaded successfully.")

    @torch.inference_mode()
    def encode(
        self,
        queries: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode single or multiple query strings into normalized vectors.

        Returns:
            np.ndarray of shape (N, 1024) in float32.
        """
        is_single = isinstance(queries, str)
        query_list = [queries] if is_single else list(queries)

        if not query_list:
            return np.empty((0, DIM), dtype=np.float32)

        embeddings_batches = []
        n_queries = len(query_list)

        for start_idx in range(0, n_queries, batch_size):
            batch_texts = query_list[start_idx : start_idx + batch_size]
            encoded_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

            outputs = self.model(**encoded_inputs)
            # CLS pooling
            emb = outputs.last_hidden_state[:, 0].float()
            # L2 normalization for cosine similarity
            emb = F.normalize(emb, p=2, dim=1)
            embeddings_batches.append(emb.cpu().numpy().astype(np.float32))

            if show_progress and n_queries > batch_size:
                done = min(start_idx + batch_size, n_queries)
                print(f"  encoded {done}/{n_queries} queries...")

        result = np.vstack(embeddings_batches)
        return result
