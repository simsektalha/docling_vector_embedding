from __future__ import annotations

import os
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from src.common.types import SearchResult
from src.ingest.upsert import VectorClient


class QdrantVectorClient(VectorClient):
    def __init__(self, url: str | None, host: str | None, port: int | None, api_key: str | None, collection: str, dims: int) -> None:
        self.collection = collection
        if url:
            self.client = QdrantClient(url=url, api_key=api_key or os.getenv("QDRANT_API_KEY"))
        else:
            self.client = QdrantClient(host=host or "localhost", port=port or 6333, api_key=api_key or os.getenv("QDRANT_API_KEY"))
        self.dims = dims

    def ensure_collection(self, name: str, dims: int) -> None:
        self.dims = dims
        has = False
        try:
            _ = self.client.get_collection(name)
            has = True
        except Exception:
            has = False
        if not has:
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            )

    def upsert(self, records):
        points = [
            PointStruct(id=r.id, vector=r.vector, payload={"text": r.text, **r.metadata})
            for r in records
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: List[float], top_k: int, filters: Optional[Dict]) -> List[SearchResult]:
        res = self.client.search(collection_name=self.collection, query_vector=query_vector, limit=top_k, query_filter=filters)
        out: List[SearchResult] = []
        for r in res:
            payload = r.payload or {}
            out.append(
                SearchResult(
                    id=str(r.id),
                    score=r.score or 0.0,
                    text=payload.get("text", ""),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )
        return out


