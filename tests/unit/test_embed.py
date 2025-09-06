from src.ingest.embed import Embedder


def test_dims_mapping_default(mocker):
    # Avoid loading real model during test
    mocker.patch("src.ingest.embed.SentenceTransformer", autospec=True)
    e = Embedder(provider="huggingface", model="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, timeout_s=10)
    assert e.embedding_dimensions() == 384


