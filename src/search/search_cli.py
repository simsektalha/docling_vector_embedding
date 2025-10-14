import argparse
import os

from src.common.config import load_config
from src.ingest.embed import Embedder
from src.ingest.upsert import make_vector_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    embedder = Embedder(
        provider=cfg["embeddings"]["provider"],
        model=cfg["embeddings"]["model"],
        batch_size=cfg["embeddings"].get("batch_size", 64),
        timeout_s=cfg["embeddings"].get("request_timeout_s", 60),
        base_url=cfg.get("embedding", {}).get("base_url"),
        api_key=cfg.get("embedding", {}).get("api_key"),
    )
    # Prefer configured dimension from pg settings if provided
    dims = cfg.get("pg", {}).get("embedding_dim") or embedder.embedding_dimensions()
    client = make_vector_client(cfg)
    client.ensure_collection(cfg.get("vectordb", {}).get("collection", "documents"), int(dims))

    qvec = embedder.embed_texts([args.query])[0]
    results = client.search(qvec, args.top_k, cfg.get("search", {}).get("filters"))

    for r in results:
        print(f"score={r.score:.3f} id={r.id} file={r.metadata.get('file_name','')} section={r.metadata.get('section_path','')}")
        snippet = (r.text[:200] + "...") if len(r.text) > 200 else r.text
        print(snippet)
        print()


if __name__ == "__main__":
    main()


