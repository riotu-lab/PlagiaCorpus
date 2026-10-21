"""Step 6 - Assemble all queries and create the stratified 90/10 train/test split.

Combines Class 1-3 + Safe queries (step 3), the per-worker Class 4 files (step 4) and
the Class 5 file (step 5). Class 4 outputs are de-duplicated by `original_text_id` and
put in corpus order so the split is deterministic. The split is stratified by
label x type x domain with a fixed seed.

Usage:
    python scripts/06_assemble_and_split.py
"""

import argparse
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from plagiacorpus.config import (
    BASE_QUERIES_FILE,
    CLASS4_DIR,
    CLASS5_FILE,
    CORPUS_FILE,
    TEST_FILE,
    TRAIN_FILE,
)
from plagiacorpus.io import load_json, save_json


def load_class4(class4_dir: Path, corpus_order: dict[str, int]) -> list[dict]:
    """Merge worker files, keeping the first query per source passage, in corpus order."""
    by_id: dict[str, dict] = {}
    for path in sorted(class4_dir.glob("class4_worker*.json")):
        for q in load_json(path):
            by_id.setdefault(q["original_text_id"], q)
    missing = len(corpus_order) - len(by_id)
    print(f"Class 4: {len(by_id):,} queries ({missing:,} passages without a successful generation)")
    return sorted(by_id.values(), key=lambda q: corpus_order[q["original_text_id"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--base-queries", type=Path, default=BASE_QUERIES_FILE)
    parser.add_argument("--class4-dir", type=Path, default=CLASS4_DIR)
    parser.add_argument("--class5", type=Path, default=CLASS5_FILE)
    parser.add_argument("--train-out", type=Path, default=TRAIN_FILE)
    parser.add_argument("--test-out", type=Path, default=TEST_FILE)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus_order = {doc["_id"]: i for i, doc in enumerate(load_json(args.corpus))}
    queries = (
        load_json(args.base_queries)
        + load_class4(args.class4_dir, corpus_order)
        + load_json(args.class5)
    )

    strata = [f"{q['label']}_{q['type']}_{q['domain']}" for q in queries]
    train, test = train_test_split(queries, test_size=args.test_size, random_state=args.seed, stratify=strata)
    save_json(train, args.train_out)
    save_json(test, args.test_out)

    train_counts = Counter(q["type"] for q in train)
    test_counts = Counter(q["type"] for q in test)
    print(f"\n{'Type':<20}{'Train':>10}{'Test':>10}")
    for qtype in sorted(train_counts | test_counts):
        print(f"{qtype:<20}{train_counts[qtype]:>10,}{test_counts[qtype]:>10,}")
    print(f"{'Total':<20}{len(train):>10,}{len(test):>10,}")
    print(f"\nSaved {args.train_out} and {args.test_out}")


if __name__ == "__main__":
    main()
