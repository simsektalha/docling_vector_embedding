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


