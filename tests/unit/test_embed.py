import sys

import src.ingest.embed as embed_module
from src.ingest.embed import Embedder


class _DummyST:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, convert_to_numpy=False):
        return [[0.0] * 384 for _ in texts]


def test_dims_mapping_default():
    # Avoid loading real model during test by swapping the class
    embed_module.SentenceTransformer = _DummyST  # type: ignore[attr-defined]
    e = Embedder(provider="huggingface", model="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, timeout_s=10)
    assert e.embedding_dimensions() == 384


def test_openai_http_mock(monkeypatch):
    class _Resp:
        def __init__(self) -> None:
            self._json = {"data": [{"embedding": [0.1, 0.2]}]}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._json

    class _Client:
        def __init__(self, *a, **k):
            pass

        def post(self, *a, **k):
            return _Resp()

    import src.ingest.embed as mod

    monkeypatch.setattr(mod, "httpx", type("_M", (), {"Client": _Client}))
    e = Embedder(provider="openai", model="text-embedding-3-small", batch_size=4, timeout_s=5, base_url="http://dummy:8000", api_key="x")
    out = e.embed_texts(["hello"])
    assert isinstance(out, list) and isinstance(out[0], list)


