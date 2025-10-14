from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from src.common.types import Chunk, EmbeddingRecord, SearchResult


def _stable_id(source_path: str, sha256: str, chunk_index: int) -> str:
    """Generate a deterministic identifier for a chunk.

    Args:
        source_path: Original file path of the document.
        sha256: Document hash to ensure stability across runs.
        chunk_index: Index of the chunk within the document.

    Returns:
        A short hex string used as stable identifier for the chunk.
    """
    base = f"{source_path}:{sha256}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()[:24]


def build_records(chunks: List[Chunk], vectors: List[List[float]]) -> List[EmbeddingRecord]:
    """Build embedding records pairing chunks with their vectors.

    Args:
        chunks: Chunk metadata and text.
        vectors: Embedding vectors corresponding to each chunk.

    Returns:
        List of `EmbeddingRecord` suitable for upsert into the vector store.
    """
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
    """Abstract vector client interface."""

    def ensure_collection(self, name: str, dims: int) -> None:  # pragma: no cover - interface
        """Create collection/table with the given dimensions if missing."""
        raise NotImplementedError

    def upsert(self, records: List[EmbeddingRecord]) -> None:  # pragma: no cover - interface
        """Upsert a batch of embedding records."""
        raise NotImplementedError

    def search(self, query_vector: List[float], top_k: int, filters: Optional[Dict]) -> List[SearchResult]:  # pragma: no cover - interface
        """Search for nearest neighbors and return ranked results."""
        raise NotImplementedError


def make_vector_client(cfg: Dict) -> VectorClient:
    """Construct a PostgreSQL/pgvector client using configuration.

    This simplified factory only supports pgvector and ignores other providers.
    It reads DSN from `pg.dsn` if present, otherwise falls back to `vectordb.dsn`.
    """
    from src.search.client_pgvector import PgVectorClient

    # Prefer new pg.* settings; fallback to legacy vectordb.* keys for compatibility
    dsn = (
        cfg.get("pg", {}).get("dsn")
        or cfg.get("vectordb", {}).get("dsn")
        or "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    dims_val = cfg.get("pg", {}).get("embedding_dim") or cfg.get("vectordb", {}).get("dims", 0)
    dims = int(dims_val) if isinstance(dims_val, int) else (int(dims_val) if isinstance(dims_val, str) and dims_val != "auto" else 0)
    collection = cfg.get("vectordb", {}).get("collection", "documents")
    return PgVectorClient(dsn=dsn, collection=collection, dims=dims)


