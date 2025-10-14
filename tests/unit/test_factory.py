from src.ingest.upsert import make_vector_client


def test_factory_pgvector(monkeypatch):
	cfg = {
		"vectordb": {
			"collection": "documents",
			"dims": 128,
			"dsn": "postgresql://user:pass@localhost:5432/db"
		}
	}
	client = make_vector_client(cfg)
	from src.search.client_pgvector import PgVectorClient
	assert isinstance(client, PgVectorClient)
