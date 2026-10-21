"""Step 9 - True language distribution of all text records (Table 5).

Runs automatic language identification (langdetect, fixed seed) over every corpus
passage and query. Languages below 0.05% of all records are grouped as
"Other (Trace Languages)"; texts that cannot be identified are reported separately.

Usage:
    python scripts/09_language_distribution.py
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import pycountry
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm.auto import tqdm

from plagiacorpus.config import CORPUS_FILE, RESULTS_DIR, TEST_FILE, TRAIN_FILE
from plagiacorpus.io import load_json, load_queries

DetectorFactory.seed = 42
TRACE_THRESHOLD = 0.0005  # 0.05% of all records
SPECIAL_NAMES = {"zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)"}


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 5:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:  # no identifiable alphabet (numbers / symbols only)
        return "unknown"


def language_name(code: str) -> str:
    if code in SPECIAL_NAMES:
        return SPECIAL_NAMES[code]
    language = pycountry.languages.get(alpha_2=code)
    return language.name if language else code.upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--queries", type=Path, nargs="+", default=[TRAIN_FILE, TEST_FILE])
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    texts = [doc["text"] for doc in load_json(args.corpus)] + [q["text"] for q in load_queries(*args.queries)]
    counts = Counter(detect_language(t) for t in tqdm(texts, desc="Detecting languages"))
    total = len(texts)

    unknown = counts.pop("unknown", 0)
    rows, other = [], 0
    for code, n in counts.most_common():
        if n >= TRACE_THRESHOLD * total:
            rows.append((language_name(code), n))
        else:
            other += n
    if other:
        rows.append(("Other (Trace Languages)", other))
    if unknown:
        rows.append(("Unknown / Symbols Only", unknown))

    print(f"\nTable 5 - Language distribution across {total:,} text records")
    print(f"{'Language':<28}{'Count':>10}{'%':>9}")
    for name, n in rows:
        print(f"{name:<28}{n:>10,}{100 * n / total:>8.2f}%")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "table5_language_distribution.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "count", "percent"])
        writer.writerows((name, n, round(100 * n / total, 2)) for name, n in rows)
    print(f"\nTable written to {out}")


if __name__ == "__main__":
    main()
