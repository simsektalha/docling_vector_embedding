## Basic RAG (Single File)

This repository is simplified to a single-file Streamlit app using:
- PostgreSQL with pgvector
- External OpenAI-compatible endpoints for embeddings and chat completions

### Run
1) Install Python 3.10+ and dependencies:
```bash
pip install streamlit httpx psycopg[binary] pgvector
```
2) Set environment variables (adjust as needed):
```bash
set PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres
set EMBEDDING_BASE_URL=http://localhost:8000
set LLM_BASE_URL=http://localhost:8001
```
Optional:
```bash
set EMBEDDING_API_KEY=...
set LLM_API_KEY=...
set EMBEDDING_MODEL=text-embedding
set LLM_MODEL=chat-model
set EMBEDDING_DIM=1024
set RETRIEVAL_TOP_K=5
```
3) Start the app:
```bash
streamlit run app.py
```

Open the browser URL shown by Streamlit and start adding text and chatting.

### File
- `app.py`: Contains database setup, embedding/search, and the Streamlit UI.

### License
MIT


