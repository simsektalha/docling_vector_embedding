from __future__ import annotations

import os
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential_jitter


class Embedder:
    def __init__(self, provider: str, model: str, batch_size: int, timeout_s: int) -> None:
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.timeout_s = timeout_s

        if provider == "openai":
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "huggingface":
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model)
        else:
            raise ValueError(f"Unknown embeddings provider: {provider}")

    def embedding_dimensions(self) -> int:
        if self.provider == "openai":
            if self.model == "text-embedding-3-large":
                return 3072
            if self.model == "text-embedding-3-small":
                return 1536
            # Reasonable default
            return 1536
        if self.provider == "huggingface":
            # Common default for MiniLM-L6
            return 384
        raise ValueError("Unknown provider")

    def _embed_batch_openai(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts, timeout=self.timeout_s)
        return [d.embedding for d in resp.data]

    def _embed_batch_hf(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=False)  # type: ignore

    @retry(wait=wait_exponential_jitter(initial=1, max=20), stop=stop_after_attempt(5))
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if self.provider == "openai":
                vectors.extend(self._embed_batch_openai(batch))
            else:
                vectors.extend(self._embed_batch_hf(batch))
        return vectors


