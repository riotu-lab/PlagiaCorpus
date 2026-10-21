"""Step 3 - Merge the Wikipedia and PD components into the bilingual source corpus.

Assigns stable identifiers (WIKI_000000, ..., PD_000000, ...) to source passages,
writes the final corpus file, and resolves every Class 1-3 / Safe query to its
`original_text_id` and `domain`.

Usage:
    python scripts/03_merge_components.py
"""

import argparse
from pathlib import Path

from plagiacorpus.config import (
    BASE_QUERIES_FILE,
    COMPONENTS,
    CORPUS_FILE,
    PD_COMPONENT_FILE,
    UNKNOWN_ID,
    WIKI_COMPONENT_FILE,
)
from plagiacorpus.io import load_json, save_json


def merge_component(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (corpus records, query records) of one component in the final schema."""
    data = load_json(path)
    info = COMPONENTS[data["component"]]
    ids = [f"{info['prefix']}_{i:06d}" for i in range(len(data["corpus"]))]

    corpus = [
        {"_id": doc_id, "text": doc["text"], "domain": info["domain"], "source": doc["source"]}
        for doc_id, doc in zip(ids, data["corpus"], strict=True)
    ]
    queries = []
    for q in data["queries"]:
        index = q["corpus_index"]
        queries.append({
            "text": q["text"],
            "type": q["type"],
            "label": q["label"],
            "source": q["source"],
            "domain": info["domain"],
            "generator_model": q["generator_model"],
            "original_text_id": UNKNOWN_ID if index is None else ids[index],
        })
    return corpus, queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--components", type=Path, nargs="+", default=[WIKI_COMPONENT_FILE, PD_COMPONENT_FILE])
    parser.add_argument("--corpus-out", type=Path, default=CORPUS_FILE)
    parser.add_argument("--queries-out", type=Path, default=BASE_QUERIES_FILE)
    args = parser.parse_args()

    corpus, queries = [], []
    for path in args.components:
        component_corpus, component_queries = merge_component(path)
        corpus += component_corpus
        queries += component_queries
        print(f"{path}: {len(component_corpus):,} passages, {len(component_queries):,} queries")

    save_json(corpus, args.corpus_out)
    save_json(queries, args.queries_out)
    print(f"Corpus : {len(corpus):,} passages -> {args.corpus_out}")
    print(f"Queries: {len(queries):,} (Class 1-3 + Safe) -> {args.queries_out}")


if __name__ == "__main__":
    main()
