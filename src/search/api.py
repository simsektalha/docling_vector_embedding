from fastapi import FastAPI
from pydantic import BaseModel

from src.common.config import load_config
from src.ingest.embed import Embedder
from src.search.client_qdrant import QdrantVectorClient


app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict | None = None


class RagRequest(BaseModel):
    query: str
    top_k: int = 5


import os
_cfg_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
_cfg = load_config(_cfg_path)
_embedder = Embedder(
    provider=_cfg["embeddings"]["provider"],
    model=_cfg["embeddings"]["model"],
    batch_size=_cfg["embeddings"].get("batch_size", 64),
    timeout_s=_cfg["embeddings"].get("request_timeout_s", 60),
)
_dims = _embedder.embedding_dimensions()
_vcfg = _cfg["vectordb"]
_client = QdrantVectorClient(_vcfg.get("url"), _vcfg.get("host"), _vcfg.get("port"), _vcfg.get("api_key"), _vcfg["collection"], _dims)
_client.ensure_collection(_vcfg["collection"], _dims)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    qvec = _embedder.embed_texts([req.query])[0]
    res = _client.search(qvec, req.top_k, req.filters)
    return [
        {
            "id": r.id,
            "score": r.score,
            "text": r.text,
            "metadata": r.metadata,
        }
        for r in res
    ]


@app.post("/rag")
def rag(req: RagRequest):
    qvec = _embedder.embed_texts([req.query])[0]
    res = _client.search(qvec, req.top_k, None)
    context = "\n\n".join([r.text for r in res])[:4000]
    try:
        from openai import OpenAI

        client = OpenAI()
        completion = client.chat.completions.create(
            model=_cfg["rag"]["llm_model"],
            messages=[
                {"role": "system", "content": "Answer using only provided context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content
    except Exception:
        answer = "[LLM unavailable]"

    sources = [
        {
            "id": r.id,
            "score": r.score,
            "source_path": r.metadata.get("source_path"),
            "section_path": r.metadata.get("section_path"),
        }
        for r in res
    ]
    return {"answer": answer, "sources": sources}


