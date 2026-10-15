"""Pure LlamaIndex PostgreSQL Vector Index connector."""

import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore

from pure_llamaindex.config import init_settings

_INDEX_CACHE = None


def get_vector_index(
    host: str = "127.0.0.1",
    port: str = "5433",
    database: str = "erag",
    user: str = "erag",
    password: str = "erag",
    table_name: str = "pure_llamaindex_chunks",
    embed_dim: int = 1024,
) -> VectorStoreIndex:
    """Connect to PostgreSQL pgvector using official LlamaIndex PGVectorStore."""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    init_settings()

    host = os.environ.get("PGHOST", host)
    port = os.environ.get("PGPORT", port)
    database = os.environ.get("PGDATABASE", database)
    user = os.environ.get("PGUSER", user)
    password = os.environ.get("PGPASSWORD", password)

    vector_store = PGVectorStore.from_params(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        table_name=table_name,
        embed_dim=embed_dim,
    )

    _INDEX_CACHE = VectorStoreIndex.from_vector_store(vector_store)
    return _INDEX_CACHE

