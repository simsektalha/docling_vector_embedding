from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from src.common.types import Chunk, EmbeddingRecord, SearchResult


def _stable_id(source_path: str, sha256: str, chunk_index: int) -> str:
    base = f"{source_path}:{sha256}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()[:24]


def build_records(chunks: List[Chunk], vectors: List[List[float]]) -> List[EmbeddingRecord]:
    assert len(chunks) == len(vectors)
    records: List[EmbeddingRecord] = []
    for ch, vec in zip(chunks, vectors):
        source_path = ch.metadata.get("source_path", "unknown")
        sha256 = ch.metadata.get("sha256", ch.doc_id)
        rid = _stable_id(source_path, sha256, ch.chunk_index)
        metadata = dict(ch.metadata)
        metadata.update(
            {
                "chunk_index": ch.chunk_index,
                "section_path": ch.section_path,
                "page_numbers": ch.page_numbers,
                "char_span": ch.char_span,
            }
        )
        records.append(EmbeddingRecord(id=rid, vector=vec, text=ch.text, metadata=metadata))
    return records


class VectorClient:
    def ensure_collection(self, name: str, dims: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def upsert(self, records: List[EmbeddingRecord]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def search(self, query_vector: List[float], top_k: int, filters: Optional[Dict]) -> List[SearchResult]:  # pragma: no cover - interface
        raise NotImplementedError


def make_vector_client(cfg: Dict) -> VectorClient:
    provider = cfg["vectordb"]["provider"]
    dims = cfg["vectordb"].get("dims", "auto")
    if provider == "qdrant":
        from src.search.client_qdrant import QdrantVectorClient

        # Caller should set dims from embedder when creating collection
        vcfg = cfg["vectordb"]
        return QdrantVectorClient(vcfg.get("url"), vcfg.get("host"), vcfg.get("port"), vcfg.get("api_key"), vcfg["collection"], int(dims) if dims != "auto" else 0)
    if provider == "pgvector":
        from src.search.client_pgvector import PgVectorClient

        vcfg = cfg["vectordb"]
        dsn = vcfg.get("dsn") or "postgresql://postgres:postgres@localhost:5432/postgres"
        return PgVectorClient(dsn=dsn, collection=vcfg["collection"], dims=int(dims) if dims != "auto" else 0)
    raise ValueError(f"Unknown vectordb provider: {provider}")


