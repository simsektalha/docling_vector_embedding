from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import psycopg
import streamlit as st
from pgvector.psycopg import register_vector


def _env(name: str, default: Optional[str] = None) -> str:
    """Fetch an environment variable with a fallback default.

    Args:
        name: Environment variable name.
        default: Default to return if the variable is missing.

    Returns:
        The environment variable value or the provided default.

    Raises:
        RuntimeError: If the variable is missing and no default is provided.
    """
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


PG_DSN: str = _env("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
PG_SCHEMA: str = _env("PG_SCHEMA", "public")
PG_TABLE: str = _env("PG_TABLE", "documents")
EMBEDDING_BASE_URL: str = _env("EMBEDDING_BASE_URL", "http://localhost:8000")
EMBEDDING_API_KEY: str = _env("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "text-embedding")
EMBEDDING_DIM: int = int(_env("EMBEDDING_DIM", "1024"))
LLM_BASE_URL: str = _env("LLM_BASE_URL", "http://localhost:8001")
LLM_API_KEY: str = _env("LLM_API_KEY", "")
LLM_MODEL: str = _env("LLM_MODEL", "chat-model")
RETRIEVAL_TOP_K: int = int(_env("RETRIEVAL_TOP_K", "5"))


@dataclass
class SearchResult:
    """A single search hit with similarity score and metadata."""

    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class PgVectorClient:
    """Minimal pgvector client with auto-provision, upsert and cosine ANN search."""

    def __init__(self, dsn: str, schema: str, table: str, dims: int) -> None:
        """Initialize and register pgvector type.

        Args:
            dsn: PostgreSQL DSN.
            schema: Target schema name.
            table: Target table name.
            dims: Embedding vector dimension.
        """
        self.dsn = dsn
        self.schema = schema
        self.table = table
        self.dims = dims
        self._conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(self._conn)

    def ensure_ready(self) -> None:
        """Create the extension, table, and IVFFLAT cosine index if not exist."""
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE SCHEMA IF NOT EXISTS {self.schema};
        CREATE TABLE IF NOT EXISTS {self.schema}.{self.table} (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            vector VECTOR({self.dims}) NOT NULL,
            source TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_{self.table}_vec
          ON {self.schema}.{self.table} USING ivfflat (vector vector_cosine_ops)
          WITH (lists = 100);
        """
        with self._conn.cursor() as cur:
            cur.execute(ddl)

    def upsert(self, records: Iterable[Tuple[str, str, List[float], Optional[str]]]) -> None:
        """Upsert embedding records into the table.

        Args:
            records: Iterable of (id, text, vector, source)
        """
        sql = f"""
        INSERT INTO {self.schema}.{self.table} (id, text, vector, source)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
          SET text=EXCLUDED.text,
              vector=EXCLUDED.vector,
              source=EXCLUDED.source;
        """
        with self._conn.cursor() as cur:
            for rid, text, vec, src in records:
                cur.execute(sql, (rid, text, vec, src))

    def search(self, query_vector: List[float], top_k: int) -> List[SearchResult]:
        """Return top_k nearest neighbors using cosine distance.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            Ranked list of search results.
        """
        sql = f"""
        SELECT id, text, source, (1 - (vector <#> %s)) AS similarity
        FROM {self.schema}.{self.table}
        ORDER BY vector <#> %s
        LIMIT {top_k}
        """
        params: List[Any] = [query_vector, query_vector]
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        results: List[SearchResult] = []
        for rid, text, source, sim in rows:
            results.append(
                SearchResult(
                    id=str(rid),
                    score=float(sim) if sim is not None else 0.0,
                    text=str(text),
                    metadata={"source": source},
                )
            )
        return results


class EmbeddingsClient:
    """HTTP client for OpenAI-compatible embeddings API."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: int = 30) -> None:
        """Initialize client with base_url and optional Bearer token."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, headers=headers)
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for the given list of texts."""
        resp = self._client.post("/v1/embeddings", json={"model": self._model, "input": texts})
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [row["embedding"] for row in data]


class ChatClient:
    """HTTP client for OpenAI-compatible chat completions API."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: int = 60) -> None:
        """Initialize client with base_url and optional Bearer token."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, headers=headers)
        self._model = model

    def complete(self, prompt: str) -> str:
        """Return a completion string for the provided prompt."""
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
        }
        resp = self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        j = resp.json()
        return j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


def assemble_prompt(contexts: List[SearchResult], question: str, max_context_chars: int = 8000) -> str:
    """Build a simple prompt from retrieved contexts plus the user question.

    Args:
        contexts: Retrieved passages.
        question: User question.
        max_context_chars: Max characters for the context block.

    Returns:
        A concise prompt string.
    """
    pieces: List[str] = []
    total = 0
    for r in contexts:
        snippet = r.text[: min(len(r.text), 800)]
        if total + len(snippet) > max_context_chars:
            break
        pieces.append(f"- {snippet}")
        total += len(snippet)
    ctx_block = "\n".join(pieces) if pieces else "No relevant context found."
    return (
        "You are a helpful assistant. Use the provided context to answer the question.\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and cite key facts."
    )


def _bootstrap_clients() -> Tuple[PgVectorClient, EmbeddingsClient, ChatClient]:
    """Create and return database, embeddings, and chat clients."""
    db = PgVectorClient(dsn=PG_DSN, schema=PG_SCHEMA, table=PG_TABLE, dims=EMBEDDING_DIM)
    db.ensure_ready()
    embedder = EmbeddingsClient(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY, model=EMBEDDING_MODEL)
    chatter = ChatClient(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL)
    return db, embedder, chatter


def _render_sources(sources: List[SearchResult]) -> None:
    """Render sources with scores in an expandable block."""
    if not sources:
        return
    with st.expander("Sources"):
        for r in sources:
            st.caption(f"{r.metadata.get('source', '')} · score={r.score:.3f}")


def _ingest_text(db: PgVectorClient, embedder: EmbeddingsClient, text: str, source: Optional[str]) -> None:
    """Embed and upsert a single text chunk.

    Args:
        db: Database client.
        embedder: Embeddings client.
        text: Text to index.
        source: Optional source label.
    """
    vectors = embedder.embed([text])
    rid = str(uuid.uuid4())
    db.upsert([(rid, text, vectors[0], source)])


def main() -> None:
    """Run the minimal Streamlit RAG application."""
    st.set_page_config(page_title="Basic RAG", page_icon="📄", layout="centered")
    st.title("Basic RAG (Postgres + External Models)")

    with st.sidebar:
        st.subheader("Settings")
        st.text_input("PG_DSN", PG_DSN, disabled=True)
        st.text_input("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL, disabled=True)
        st.text_input("LLM_BASE_URL", LLM_BASE_URL, disabled=True)
        top_k = st.slider("Top K", min_value=1, max_value=10, value=RETRIEVAL_TOP_K)

    try:
        db, embedder, chatter = _bootstrap_clients()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return

    st.header("Add a document")
    with st.form("ingest_form"):
        src_text = st.text_input("Source (optional)", value="")
        doc_text = st.text_area("Paste text to index", height=160, placeholder="Paste any text here...")
        submitted = st.form_submit_button("Add")
        if submitted:
            try:
                txt = (doc_text or "").strip()
                if not txt:
                    st.warning("Please paste some text.")
                else:
                    _ingest_text(db, embedder, txt, src_text or None)
                    st.success("Document added.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.header("Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{'role': 'user'|'assistant', 'content': str}]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                try:
                    qvec = embedder.embed([prompt])[0]
                    hits = db.search(qvec, top_k=top_k)
                    full_prompt = assemble_prompt(hits, prompt, max_context_chars=8000)
                    completion = chatter.complete(full_prompt)
                    st.markdown(completion or "[No answer]")
                    _render_sources(hits)
                    st.session_state.messages.append({"role": "assistant", "content": completion or "[No answer]"})
                except Exception as e:
                    st.error(f"Chat error: {e}")


if __name__ == "__main__":
    main()


