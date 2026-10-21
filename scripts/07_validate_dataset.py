"""Step 7 - Structural validation of the released files (Technical Validation).

Checks that every query is consistent with the passage referenced by its
`original_text_id`:

  errors    unique corpus ids; required fields; label <-> type; Safe queries use
            UNKNOWN_ID; positive queries resolve to a corpus passage; Class 1 equals its
            source; Class 2 equals the obfuscated source; Class 5 source and query
            languages match the translation direction.
  warnings  duplicate corpus passages; Class 3/4 queries identical to their source or
            in a different language from it.

Exits with status 1 if any error is found.

Usage:
    python scripts/07_validate_dataset.py
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from plagiacorpus.config import CORPUS_FILE, TEST_FILE, TRAIN_FILE, UNKNOWN_ID
from plagiacorpus.io import load_json, load_queries
from plagiacorpus.text import language_of, obfuscate

REQUIRED_FIELDS = ("text", "type", "label", "source", "domain", "generator_model", "original_text_id")

ERRORS = [
    "corpus: duplicate _id",
    "query: missing field(s)",
    "query: label inconsistent with type",
    "Safe: original_text_id is not UNKNOWN_ID",
    "positive: original_text_id not found in corpus",
    "Class 1: text differs from source passage",
    "Class 2: text is not the obfuscated source passage",
    "Class 5: source passage language does not match direction",
    "Class 5: query language does not match direction",
]
WARNINGS = [
    "corpus: duplicate passage text",
    "Class 3/4: query identical to source passage",
    "Class 3/4: query language differs from source passage",
]


def describe(q: dict) -> str:
    return f"{q.get('type')} | {q.get('original_text_id')} | {q.get('text', '')[:70]!r}"


def validate(corpus: list[dict], queries: list[dict]) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = defaultdict(list)

    id_counts = Counter(doc["_id"] for doc in corpus)
    issues["corpus: duplicate _id"] = [i for i, n in id_counts.items() if n > 1]
    text_counts = Counter(doc["text"] for doc in corpus)
    issues["corpus: duplicate passage text"] = [
        doc["_id"] for doc in corpus if text_counts[doc["text"]] > 1
    ]
    passages = {doc["_id"]: doc["text"] for doc in corpus}

    for q in queries:
        if any(field not in q for field in REQUIRED_FIELDS):
            issues["query: missing field(s)"].append(describe(q))
            continue
        qtype, oid = q["type"], q["original_text_id"]
        cls = qtype.split(" (")[0]
        if (q["label"] == 0) != (cls == "Safe"):
            issues["query: label inconsistent with type"].append(describe(q))
        if cls == "Safe":
            if oid != UNKNOWN_ID:
                issues["Safe: original_text_id is not UNKNOWN_ID"].append(describe(q))
            continue
        if oid not in passages:
            issues["positive: original_text_id not found in corpus"].append(describe(q))
            continue

        source, text = passages[oid], q["text"]
        if cls == "Class 1" and text != source:
            issues["Class 1: text differs from source passage"].append(describe(q))
        elif cls == "Class 2" and text != obfuscate(source, "ar" if "(AR" in qtype else "en"):
            issues["Class 2: text is not the obfuscated source passage"].append(describe(q))
        elif cls in ("Class 3", "Class 4"):
            if text.strip() == source.strip():
                issues["Class 3/4: query identical to source passage"].append(describe(q))
            if language_of(text) != language_of(source):
                issues["Class 3/4: query language differs from source passage"].append(describe(q))
        elif cls == "Class 5":
            src_lang, tgt_lang = ("en", "ar") if "EN->AR" in qtype else ("ar", "en")
            if language_of(source) != src_lang:
                issues["Class 5: source passage language does not match direction"].append(describe(q))
            if language_of(text) != tgt_lang:
                issues["Class 5: query language does not match direction"].append(describe(q))
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--queries", type=Path, nargs="+", default=[TRAIN_FILE, TEST_FILE])
    parser.add_argument("--examples", type=int, default=3, help="Examples to print per failed check.")
    args = parser.parse_args()

    corpus = load_json(args.corpus)
    queries = load_queries(*args.queries)
    print(f"Validating {len(corpus):,} passages and {len(queries):,} queries\n")
    issues = validate(corpus, queries)

    n_errors = 0
    for severity, checks in (("ERROR", ERRORS), ("WARNING", WARNINGS)):
        for check in checks:
            found = issues.get(check, [])
            print(f"[{'ok' if not found else severity}] {check}: {len(found):,}")
            for example in found[: args.examples]:
                print(f"      {example}")
            if severity == "ERROR":
                n_errors += len(found)

    print(f"\n{'PASSED' if n_errors == 0 else 'FAILED'}: {n_errors:,} error(s)")
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
