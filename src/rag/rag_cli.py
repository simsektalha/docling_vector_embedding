import argparse

from src.rag.retrieve import retrieve
from src.rag.generate import answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results = retrieve(args.query, args.config, args.top_k, None)
    res = answer(args.query, results, args.config)
    print(res["answer"])
    print("\nSources:")
    for s in res["sources"]:
        print(f"- {s['source_path']} ({s['section_path']}) score={s['score']:.3f}")


if __name__ == "__main__":
    main()


