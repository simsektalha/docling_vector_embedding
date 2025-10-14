from typing import Dict, List, Optional

from src.common.config import load_config
from src.ingest.embed import Embedder
from src.ingest.upsert import make_vector_client
from src.common.types import SearchResult


def retrieve(query: str, cfg_path: str, top_k: int, filters: Optional[Dict]) -> List[SearchResult]:
    cfg = load_config(cfg_path)
    embedder = Embedder(
        provider=cfg["embeddings"]["provider"],
        model=cfg["embeddings"]["model"],
        batch_size=cfg["embeddings"].get("batch_size", 64),
        timeout_s=cfg["embeddings"].get("request_timeout_s", 60),
        base_url=cfg.get("embedding", {}).get("base_url"),
        api_key=cfg.get("embedding", {}).get("api_key"),
    )
    dims = cfg.get("pg", {}).get("embedding_dim") or embedder.embedding_dimensions()
    client = make_vector_client(cfg)
    client.ensure_collection(cfg.get("vectordb", {}).get("collection", "documents"), int(dims))
    qvec = embedder.embed_texts([query])[0]
    return client.search(qvec, top_k, filters)


