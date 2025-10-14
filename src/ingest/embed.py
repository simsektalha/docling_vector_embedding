from __future__ import annotations

import os
from typing import List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential


class Embedder:
    """Embeddings client supporting OpenAI SDK, OpenAI-compatible HTTP, and HF local models.

    If `base_url` is provided for provider="openai", the client will call the
    OpenAI-compatible HTTP endpoint at `{base_url}/v1/embeddings` using `httpx`.
    Otherwise it will use the official OpenAI SDK.
    """

    def __init__(self, provider: str, model: str, batch_size: int, timeout_s: int, base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self._http_base_url = base_url
        self._api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")

        if provider == "openai":
            if base_url:
                # Use HTTP transport to external OpenAI-compatible service
                self._http_client = httpx.Client(base_url=base_url, timeout=timeout_s, headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else None)
                self._use_http = True
            else:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self._api_key)
                self._use_http = False
        elif provider == "huggingface":
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model)
            self._use_http = False
        else:
            raise ValueError(f"Unknown embeddings provider: {provider}")

    def embedding_dimensions(self) -> int:
        if self.provider == "openai":
            if self.model == "text-embedding-3-large":
                return 3072
            if self.model == "text-embedding-3-small":
                return 1536
            return 1536
        if self.provider == "huggingface":
            return 384
        raise ValueError("Unknown provider")

    def _embed_batch_openai(self, texts: List[str]) -> List[List[float]]:
        if getattr(self, "_use_http", False):
            resp = self._http_client.post("/v1/embeddings", json={"model": self.model, "input": texts})
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data.get("data", [])]
        resp = self._client.embeddings.create(model=self.model, input=texts, timeout=self.timeout_s)  # type: ignore[attr-defined]
        return [d.embedding for d in resp.data]

    def _embed_batch_hf(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=False)  # type: ignore[attr-defined]

    @retry(wait=wait_random_exponential(multiplier=1, max=20), stop=stop_after_attempt(5))
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if self.provider == "openai":
                vectors.extend(self._embed_batch_openai(batch))
            else:
                vectors.extend(self._embed_batch_hf(batch))
        return vectors


