"""Step 10 - Draw the query sample for human annotation (Human Annotation Reliability).

Samples 50 queries from each of the six classes (300 in total) and shuffles them. Each
row shows the query and the passage it is compared with: the source passage for Classes
1-5, or an unrelated passage from the same component and language for Safe queries.
Hide `Hidden_True_Class` from annotators for a blind review.

`--classes`, `--balance-domains` and `--first-id` draw a sample for a subset of classes,
optionally split evenly across components (used for the Class 4 items).

Usage:
    python scripts/10_sample_for_annotation.py
    python scripts/10_sample_for_annotation.py --classes "Class 4" --balance-domains
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from plagiacorpus.config import CORPUS_FILE, RESULTS_DIR, TEST_FILE, TRAIN_FILE, UNKNOWN_ID
from plagiacorpus.io import load_json, load_queries, write_table
from plagiacorpus.text import language_of

CLASSES = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Safe"]
RATING_COLUMNS = ["Is_Plagiarism", "Linguistic_Quality", "Semantic_Preservation",
                  "Obfuscation_Difficulty", "Annotator_Notes"]


def draw(pool: list[dict], n: int, rng: random.Random, balance_domains: bool) -> list[dict]:
    """Sample n queries, optionally split evenly across domains."""
    if not balance_domains:
        return rng.sample(pool, min(n, len(pool)))
    by_domain = defaultdict(list)
    for q in pool:
        by_domain[q["domain"]].append(q)
    domains = sorted(by_domain)
    sample = []
    for i, domain in enumerate(domains):
        share = n // len(domains) + (1 if i < n % len(domains) else 0)
        sample += rng.sample(by_domain[domain], min(share, len(by_domain[domain])))
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--queries", type=Path, nargs="+", default=[TRAIN_FILE, TEST_FILE])
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "annotation_sample.xlsx")
    parser.add_argument("--classes", nargs="+", default=CLASSES, choices=CLASSES)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--balance-domains", action="store_true", help="Split each class evenly across domains.")
    parser.add_argument("--first-id", type=int, default=1, help="Number of the first Sample_ID.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = load_json(args.corpus)
    passages = {doc["_id"]: doc["text"] for doc in corpus}
    same_group = defaultdict(list)  # (domain, language) -> passage ids, for Safe comparisons
    for doc in corpus:
        same_group[(doc["domain"], language_of(doc["text"]))].append(doc["_id"])

    by_class = {cls: [] for cls in args.classes}
    for q in load_queries(*args.queries):
        cls = q["type"].split(" (")[0]
        if cls in by_class:
            by_class[cls].append(q)

    rng = random.Random(args.seed)
    sample = []
    for cls, pool in by_class.items():
        drawn = draw(pool, args.per_class, rng, args.balance_domains)
        sample += drawn
        print(f"{cls}: sampled {len(drawn)} of {len(pool):,}")
    rng.shuffle(sample)

    rows = []
    for i, q in enumerate(sample, start=args.first_id):
        passage_id = q["original_text_id"]
        if passage_id == UNKNOWN_ID:  # Safe query: compare with an unrelated passage
            lang = "ar" if "(AR" in q["type"] else "en"
            passage_id = rng.choice(same_group[(q["domain"], lang)])
        rows.append({
            "Sample_ID": f"Sample_{i:03d}",
            "Hidden_True_Class": q["type"],
            "Original_Text_ID": passage_id,
            "Query_Text": q["text"],
            "Source_Text": passages[passage_id],
            **{column: "" for column in RATING_COLUMNS},
        })

    write_table(pd.DataFrame(rows), args.output)
    print(f"\n{len(rows)} samples written to {args.output}")


if __name__ == "__main__":
    main()
