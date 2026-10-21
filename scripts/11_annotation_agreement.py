"""Step 11 - Inter-annotator agreement on the annotated sample (Table 8).

Compares the two annotators' sheets item by item (matched on Sample_ID). For each dimension,
items that either annotator left blank or marked as not applicable (N/A, NA) are excluded.
The script reports N, raw agreement and Cohen's kappa. For the ordinal ratings it also
reports the quadratic weighted kappa and the share of ratings within one scale point
(Within +/-1).

Usage:
    python scripts/11_annotation_agreement.py
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from plagiacorpus.config import RESULTS_DIR
from plagiacorpus.io import read_table

MISSING = {"", "NA", "N/A"}
DIMENSIONS = {  # column -> scale type
    "Is_Plagiarism": "nominal",
    "Linguistic_Quality": "ordinal",
    "Semantic_Preservation": "ordinal",
    "Obfuscation_Difficulty": "ordinal",
}


def agreement(a: pd.Series, b: pd.Series, dimension: str, scale: str) -> dict:
    pairs = pd.concat({"a": a.str.strip(), "b": b.str.strip()}, axis=1, join="inner")
    pairs = pairs[~pairs.a.isin(MISSING) & ~pairs.b.isin(MISSING)]
    x, y = (pairs.a.astype(float), pairs.b.astype(float)) if scale == "ordinal" else (pairs.a, pairs.b)
    row = {
        "dimension": dimension,
        "scale": scale,
        "n": len(pairs),
        "agreement": (x == y).mean(),
        "kappa": cohen_kappa_score(x, y),
        "weighted_kappa": None,
        "within_1": None,
    }
    if scale == "ordinal":
        row["weighted_kappa"] = cohen_kappa_score(x, y, weights="quadratic")
        row["within_1"] = ((x - y).abs() <= 1).mean()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotator-1", type=Path, default=Path("annotation/annotator_1.xlsx"))
    parser.add_argument("--annotator-2", type=Path, default=Path("annotation/annotator_2.xlsx"))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    a = read_table(args.annotator_1).set_index("Sample_ID")
    b = read_table(args.annotator_2).set_index("Sample_ID")
    if set(a.index) != set(b.index):
        print(f"Warning: {len(set(a.index) ^ set(b.index))} Sample_IDs are rated by only one annotator.")
    print(f"Items per annotator: {len(a)}")

    rows = [agreement(a[dim], b[dim], dim, scale) for dim, scale in DIMENSIONS.items()]

    print(f"\nTable 8 - Inter-annotator agreement\n"
          f"{'Dimension':<24}{'N':>5}{'Agreement':>11}{'Kappa':>8}{'Weighted kappa':>16}{'Within +/-1':>13}")
    for r in rows:
        ordinal = r["scale"] == "ordinal"
        print(f"{r['dimension'].replace('_', ' '):<24}{r['n']:>5}{r['agreement']:>11.3f}{r['kappa']:>8.3f}"
              + (f"{r['weighted_kappa']:>16.3f}{100 * r['within_1']:>12.2f}%" if ordinal else ""))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "table8_annotation_agreement.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nTable written to {out}")


if __name__ == "__main__":
    main()
