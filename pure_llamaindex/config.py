"""Pure LlamaIndex global configuration for embeddings and LLM."""

import os
from pathlib import Path
import torch
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from generation.opencode_client import get_api_key_for_model

# Local BGE-M3 snapshot to eliminate network downloads
DEFAULT_EMBED_PATH = os.environ.get(
    "ERAG_EMBED_MODEL_PATH",
    "/home/parmpal/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
)
DEFAULT_MODEL = os.environ.get(
    "ERAG_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
)
DEFAULT_BASE_URL = os.environ.get(
    "ERAG_LLM_API_BASE", "https://integrate.api.nvidia.com/v1"
)


def init_settings():
    """Configure official LlamaIndex Settings singletons."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Pure LlamaIndex HuggingFace Embedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=DEFAULT_EMBED_PATH,
        device=dev,
    )

    # Pure LlamaIndex OpenAILike LLM
    api_key = get_api_key_for_model(DEFAULT_MODEL)
    Settings.llm = OpenAILike(
        model=DEFAULT_MODEL,
        api_base=DEFAULT_BASE_URL,
        api_key=api_key,
        is_chat_model=True,
        context_window=128000,
        timeout=120.0,
        temperature=0.0,
    )

