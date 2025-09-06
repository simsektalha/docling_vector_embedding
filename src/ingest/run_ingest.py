import argparse
import os
from pathlib import Path

from src.common.config import load_config
from src.common.logging import get_logger, setup_logging
from src.ingest.convert_docling import convert_with_docling
from src.ingest.discover import discover_files
from src.ingest.chunk import chunk_document
from src.ingest.embed import Embedder
from src.ingest.upsert import build_records, make_vector_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(level=cfg.get("logging", {}).get("level", "INFO"), json_output=cfg.get("logging", {}).get("json", True))
    log = get_logger("ingest")

    files = discover_files(
        cfg["data"]["input_dir"],
        cfg["data"].get("include_glob", []),
        cfg["data"].get("exclude_glob", []),
        cfg["data"].get("max_file_mb", 50),
    )
    log.info(f"discovered_files={len(files)}")

    embedder = Embedder(
        provider=cfg["embeddings"]["provider"],
        model=cfg["embeddings"]["model"],
        batch_size=cfg["embeddings"].get("batch_size", 64),
        timeout_s=cfg["embeddings"].get("request_timeout_s", 60),
    )
    dims = embedder.embedding_dimensions()
    client = make_vector_client(cfg)
    client.ensure_collection(cfg["vectordb"]["collection"], dims)

    total_chunks = 0
    for f in files:
        dc = convert_with_docling(f.path, f.sha256, cfg)
        chunks = chunk_document(dc, cfg["chunking"]["strategy"], cfg["chunking"]["max_tokens"], cfg["chunking"]["overlap_tokens"])
        for ch in chunks:
            ch.metadata.update(
                {
                    "source_path": f.path,
                    "file_name": Path(f.path).name,
                    "sha256": f.sha256,
                    "doc_id": dc.doc_id,
                }
            )
        vectors = embedder.embed_texts([c.text for c in chunks])
        records = build_records(chunks, vectors)
        client.upsert(records)
        total_chunks += len(chunks)

    log.info(f"ingestion_complete total_chunks={total_chunks}")


if __name__ == "__main__":
    main()


