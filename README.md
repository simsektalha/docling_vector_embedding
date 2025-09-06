## Docling RAG Starter

End-to-end, config-driven pipeline to convert documents with Docling, chunk and embed content, index into a vector database (Qdrant by default), and provide semantic search plus a minimal RAG demo with citations.

### Features
- Docling conversion (PDF, DOCX, HTML) with caching
- Hierarchical chunking w/ token fallback and overlap
- Embeddings: OpenAI by default, Sentence-Transformers fallback
- Vector DB: Qdrant by default; pluggable adapters
- CLI for ingestion, search, and RAG; minimal REST API (FastAPI)
- Config-first via YAML and .env
- Docker-compose for one-command run
- Unit tests and CI workflow

Note: Defaults use OpenAI for embeddings and LLM to provide strong quality out of the box, matching a preference for external API usage. Switch to offline HuggingFace or Ollama in config if desired.

### Quickstart
1) Prereqs
- Docker and Docker Compose
- Python 3.11+

2) Setup
- Copy `.env.example` to `.env` and fill secrets.
- Review `configs/default.yaml` and adjust if needed.

3) One-command bring-up (Qdrant + API)
```bash
docker compose up --build -d
```

4) Ingest sample docs into Qdrant
```bash
python -m venv .venv && . .venv/Scripts/activate
pip install -r requirements.txt
python samples/generate_samples.py
python -m src.ingest.run_ingest --config configs/default.yaml
```

5) Semantic search (CLI)
```bash
python -m src.search.search_cli --config configs/default.yaml --query "What is configured by default?"
```

6) Minimal RAG (CLI)
```bash
python -m src.rag.rag_cli --config configs/default.yaml --query "Summarize the key points."
```

7) REST API
- API runs at `http://localhost:8000` when `docker compose up` includes the `api` service.
- Endpoints:
  - GET `/health`
  - POST `/search` { query, top_k?, filters? }
  - POST `/rag` { query, top_k? }

### Note on dependencies
- Install `docling`, `qdrant-client`, and `tiktoken` as listed in `requirements.txt`.
- If you prefer offline embeddings, set `embeddings.provider: huggingface` and `model: sentence-transformers/all-MiniLM-L6-v2` in `configs/default.yaml`.

### Configuration
See `configs/default.yaml` for all keys. Set secrets in `.env` (e.g., `OPENAI_API_KEY`).

Key toggles:
- `embeddings.provider`: `openai` or `huggingface`
- `vectordb.provider`: `qdrant` (default), adapters ready for extension
- `chunking.strategy`: `hierarchical` or `token`

### Samples
Run `python samples/generate_samples.py` to create a tiny PDF and DOCX in `samples/docs/` for testing.

### Tests
```bash
pytest -q
```
Notes:
- Unit tests run offline.
- Integration tests are marked and require Qdrant running; they are skipped in CI by default.

### Troubleshooting
- Qdrant readiness: check `http://localhost:6333/readyz`.
- Missing OCR utilities: the Dockerfile installs Tesseract and Poppler.
- No OpenAI key: switch to `embeddings.provider: huggingface` in config.

### License
MIT


