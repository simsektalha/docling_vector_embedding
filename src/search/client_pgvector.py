from __future__ import annotations

import os
from typing import Dict, List, Optional

import psycopg
from pgvector.psycopg import register_vector

from src.common.types import SearchResult
from src.ingest.upsert import VectorClient


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector VECTOR(%(dims)s) NOT NULL,
    source_path TEXT,
    file_name TEXT,
    section_path TEXT,
    page_numbers JSONB,
    char_span JSONB,
    doc_id TEXT,
    sha256 TEXT
);
CREATE INDEX IF NOT EXISTS documents_vec_idx ON documents USING ivfflat (vector vector_l2_ops) WITH (lists = 100);
"""


class PgVectorClient(VectorClient):
    def __init__(self, dsn: str, collection: str, dims: int) -> None:
        self.dsn = dsn
        self.collection = collection  # kept for interface parity
        self.dims = dims
        self._conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(self._conn)

    def ensure_collection(self, name: str, dims: int) -> None:
        self.dims = dims
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA_SQL, {"dims": dims})

    def upsert(self, records) -> None:
        with self._conn.cursor() as cur:
            for r in records:
                payload = {**r.metadata}
                cur.execute(
                    """
                    INSERT INTO documents (id, text, vector, source_path, file_name, section_path, page_numbers, char_span, doc_id, sha256)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, vector=EXCLUDED.vector,
                    source_path=EXCLUDED.source_path, file_name=EXCLUDED.file_name, section_path=EXCLUDED.section_path,
                    page_numbers=EXCLUDED.page_numbers, char_span=EXCLUDED.char_span, doc_id=EXCLUDED.doc_id, sha256=EXCLUDED.sha256
                    """,
                    (
                        r.id,
                        r.text,
                        r.vector,
                        payload.get("source_path"),
                        payload.get("file_name"),
                        payload.get("section_path"),
                        payload.get("page_numbers"),
                        payload.get("char_span"),
                        payload.get("doc_id"),
                        payload.get("sha256"),
                    ),
                )

    def search(self, query_vector: List[float], top_k: int, filters: Optional[Dict]) -> List[SearchResult]:
        where = []
        params: List = [query_vector]
        if filters:
            for k, v in filters.items():
                where.append(f"{k} = %s")
                params.append(v)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = f"SELECT id, text, source_path, file_name, section_path, page_numbers, char_span, doc_id, sha256, (vector <-> %s) AS distance FROM documents{where_sql} ORDER BY vector <-> %s LIMIT {top_k}"
        # duplicate query_vector at end for ORDER BY
        params.append(query_vector)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out: List[SearchResult] = []
        for row in rows:
            (rid, text, source_path, file_name, section_path, page_numbers, char_span, doc_id, sha256, distance) = row
            out.append(
                SearchResult(
                    id=rid,
                    score=float(1.0 / (1.0 + distance)) if distance is not None else 0.0,
                    text=text,
                    metadata={
                        "source_path": source_path,
                        "file_name": file_name,
                        "section_path": section_path,
                        "page_numbers": page_numbers,
                        "char_span": char_span,
                        "doc_id": doc_id,
                        "sha256": sha256,
                    },
                )
            )
        return out


