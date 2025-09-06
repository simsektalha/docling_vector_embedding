import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config and apply environment overrides.

    Environment variables can override selected keys, e.g.,
    - OPENAI_API_KEY (used by providers directly)
    - QDRANT_URL / QDRANT_API_KEY will override vectordb.url/api_key if present
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Inject env overrides for convenience
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        cfg.setdefault("vectordb", {})["url"] = qdrant_url
    # DSN override for pgvector
    dsn = os.getenv("VECTOR_DSN") or os.getenv("PGVECTOR_DSN")
    if dsn:
        cfg.setdefault("vectordb", {})["dsn"] = dsn
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_api_key is not None:
        cfg.setdefault("vectordb", {})["api_key"] = qdrant_api_key

    # Embeddings provider/model can be overridden
    embed_provider = os.getenv("EMBEDDINGS_PROVIDER")
    if embed_provider:
        cfg.setdefault("embeddings", {})["provider"] = embed_provider
    embed_model = os.getenv("EMBEDDINGS_MODEL")
    if embed_model:
        cfg.setdefault("embeddings", {})["model"] = embed_model

    # LLM overrides
    llm_provider = os.getenv("LLM_PROVIDER")
    if llm_provider:
        cfg.setdefault("rag", {})["llm_provider"] = llm_provider
    llm_model = os.getenv("LLM_MODEL")
    if llm_model:
        cfg.setdefault("rag", {})["llm_model"] = llm_model

    return cfg


