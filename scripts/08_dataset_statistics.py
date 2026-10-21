"""Step 8 - Dataset statistics (Data Records and Technical Validation).

Reports corpus size and word counts, the provenance of source passages (Table 2),
the train/test and label distributions, and per-class counts, dominant generator
model and word-length statistics (Table 7). Tables are also written as CSV files.

Usage:
    python scripts/08_dataset_statistics.py
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from plagiacorpus.config import CORPUS_FILE, RESULTS_DIR, TEST_FILE, TRAIN_FILE
from plagiacorpus.io import load_json
from plagiacorpus.text import count_words, language_of

WIKI_HOST = re.compile(r"https?://([a-z]+)\.wikipedia\.org")
WIKI_LANGUAGE = {"en": "English", "ar": "Arabic"}


def repository(doc: dict) -> str:
    """Primary repository of a corpus passage, derived from its `source` field."""
    source = doc.get("source", "")
    if source.startswith("HF://"):
        return "HF://" + "/".join(source[len("HF://"):].split("/")[:2])
    host = WIKI_HOST.match(source)
    if host:
        return f"{WIKI_LANGUAGE.get(host.group(1), host.group(1))} Wikipedia"
    component = doc.get("domain") or doc["_id"].split("_")[0]
    return f"{component} ({language_of(doc['text']).upper()})"


def length_stats(lengths: list[int]) -> dict:
    a = np.asarray(lengths)
    return {"total": int(a.sum()), "mean": a.mean(), "median": np.median(a), "std": a.std(),
            "min": int(a.min()), "max": int(a.max())}


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    corpus = load_json(args.corpus)
    train, test = load_json(args.train), load_json(args.test)
    queries = train + test

    # --- Source corpus -----------------------------------------------------
    s = length_stats([count_words(doc["text"]) for doc in corpus])
    print(f"Source corpus: {len(corpus):,} passages, {s['total']:,} words "
          f"(mean {s['mean']:.1f}, median {s['median']:.0f}, std {s['std']:.1f}, "
          f"min {s['min']}, max {s['max']:,})")

    # --- Table 2: provenance --------------------------------------------------
    provenance = Counter(repository(doc) for doc in corpus)
    table2 = [{"repository": repo, "count": n, "percent": round(100 * n / len(corpus), 2)}
              for repo, n in provenance.most_common()]
    print(f"\nTable 2 - Data provenance\n{'Repository':<40}{'Count':>10}{'%':>9}")
    for row in table2:
        print(f"{row['repository']:<40}{row['count']:>10,}{row['percent']:>8.2f}%")

    # --- Query overview -------------------------------------------------------
    labels = Counter(q["label"] for q in queries)
    print(f"\nQueries: {len(queries):,} (train {len(train):,}, test {len(test):,})")
    print(f"  plagiarised (label=1): {labels[1]:,} ({100 * labels[1] / len(queries):.1f}%)")
    print(f"  safe        (label=0): {labels[0]:,} ({100 * labels[0] / len(queries):.1f}%)")
    print(f"Total text records (corpus + queries): {len(corpus) + len(queries):,}")

    # --- Table 7: per-class statistics ----------------------------------------
    lengths, models = defaultdict(list), defaultdict(Counter)
    for q in queries:
        lengths[q["type"]].append(count_words(q["text"]))
        models[q["type"]][q["generator_model"]] += 1

    table7 = []
    for qtype in sorted(lengths):
        st = length_stats(lengths[qtype])
        table7.append({
            "class": qtype,
            "count": len(lengths[qtype]),
            "percent": round(100 * len(lengths[qtype]) / len(queries), 1),
            "dominant_model": models[qtype].most_common(1)[0][0],
            "mean_words": round(st["mean"], 1),
            "median_words": round(st["median"], 1),
            "std_words": round(st["std"], 1),
        })
    print(f"\nTable 7 - Word-length statistics per class\n"
          f"{'Class':<20}{'Count':>9}{'%':>7}  {'Dominant model':<34}{'Mean':>8}{'Median':>8}{'Std':>8}")
    for r in table7:
        print(f"{r['class']:<20}{r['count']:>9,}{r['percent']:>6.1f}%  {r['dominant_model']:<34}"
              f"{r['mean_words']:>8.1f}{r['median_words']:>8.0f}{r['std_words']:>8.1f}")

    write_csv(table2, args.out_dir / "table2_provenance.csv")
    write_csv(table7, args.out_dir / "table7_class_statistics.csv")
    print(f"\nTables written to {args.out_dir}/")


if __name__ == "__main__":
    main()
