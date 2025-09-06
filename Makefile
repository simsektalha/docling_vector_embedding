CONFIG?=configs/default.yaml
QUERY?=What is configured by default?

.PHONY: up down install ingest search rag api test samples

up:
	docker compose up -d qdrant

down:
	docker compose down

install:
	python -m venv .venv && . .venv/Scripts/activate && pip install -r requirements.txt

ingest:
	python -m src.ingest.run_ingest --config $(CONFIG)

search:
	python -m src.search.search_cli --config $(CONFIG) --query "$(QUERY)"

rag:
	python -m src.rag.rag_cli --config $(CONFIG) --query "$(QUERY)"

api:
	docker compose up --build api

samples:
	python samples/generate_samples.py

test:
	pytest -q


