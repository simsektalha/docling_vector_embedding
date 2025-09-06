CONFIG?=configs/default.yaml
QUERY?=What is configured by default?

.PHONY: up down install ingest search rag api test samples eval

up:
	docker compose up -d postgres

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

eval:
	python -m src.eval --config $(CONFIG) --queries samples/queries.yaml --k 5


