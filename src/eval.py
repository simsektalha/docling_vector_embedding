import argparse
import yaml

from src.rag.retrieve import retrieve


def hit_at_k(expected_terms, results):
    hay = "\n".join([r.text for r in results]).lower()
    return any(term.lower() in hay for term in expected_terms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", default="samples/queries.yaml")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        queries = yaml.safe_load(f)

    hits = 0
    for q in queries:
        results = retrieve(q["query"], args.config, args.k, None)
        if hit_at_k(q.get("expects_any_source_contains", []), results):
            hits += 1
    print(f"hit@{args.k}={hits}/{len(queries)}")


if __name__ == "__main__":
    main()


