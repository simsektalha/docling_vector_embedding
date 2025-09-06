import argparse
import os

from src.common.config import load_config
from src.ingest.embed import Embedder
from src.search.client_qdrant import QdrantVectorClient


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
    )
    dims = embedder.embedding_dimensions()
    vcfg = cfg["vectordb"]
    client = QdrantVectorClient(vcfg.get("url"), vcfg.get("host"), vcfg.get("port"), vcfg.get("api_key"), vcfg["collection"], dims)
    client.ensure_collection(vcfg["collection"], dims)

    qvec = embedder.embed_texts([args.query])[0]
    results = client.search(qvec, args.top_k, cfg.get("search", {}).get("filters"))

    for r in results:
        print(f"score={r.score:.3f} id={r.id} file={r.metadata.get('file_name','')} section={r.metadata.get('section_path','')}")
        snippet = (r.text[:200] + "...") if len(r.text) > 200 else r.text
        print(snippet)
        print()


if __name__ == "__main__":
    main()


