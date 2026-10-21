"""Step 2 - Public-domain (PD) component.

Streams English (PleIAs/English-PD) and Arabic (mohres/The_Arabic_E-Book_Corpus)
paragraphs (> 150 characters) from the Hugging Face Hub, takes independent Safe
control pools from later portions of the same collections, and generates Class 1,
Class 2, Class 3 (Qwen2.5-14B-Instruct) and Safe queries.

Usage:
    python scripts/02_build_public_domain_component.py
"""

import argparse
from pathlib import Path

from plagiacorpus.component import build_component, load_or_build_pools, make_class3_paraphraser
from plagiacorpus.config import PD_COMPONENT_FILE, PD_DATASETS

MIN_PARAGRAPH_CHARS = 150


def stream_paragraphs(dataset_id: str, count: int, skip: int = 0) -> list[dict]:
    """Return `count` paragraphs from a streamed HF dataset after skipping the first `skip`."""
    from datasets import load_dataset

    passages: list[dict] = []
    skipped = 0
    for record in load_dataset(dataset_id, split="train", streaming=True):
        text = record.get("text") or record.get("content") or ""
        title = record.get("title") or record.get("booknr") or "Public Domain Book"
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if len(paragraph) <= MIN_PARAGRAPH_CHARS:
                continue
            if skipped < skip:
                skipped += 1
                continue
            passages.append({"text": paragraph, "url": f"HF://{dataset_id}/{title}"})
            if len(passages) >= count:
                return passages
    return passages


def collect_pools(per_lang: int) -> dict:
    # Safe passages come after the first 2 * per_lang paragraphs, so they never overlap the sources.
    return {
        "sources": {lang: stream_paragraphs(ds, per_lang) for lang, ds in PD_DATASETS.items()},
        "safe": {lang: stream_paragraphs(ds, per_lang, skip=2 * per_lang) for lang, ds in PD_DATASETS.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=PD_COMPONENT_FILE)
    parser.add_argument("--paragraphs-per-lang", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    pools_file = args.output.with_name(args.output.stem + "_pools.json")
    pools = load_or_build_pools(pools_file, lambda: collect_pools(args.paragraphs_per_lang))

    def pick_safe(_idx: int, lang_idx: int, lang: str):
        pool = pools["safe"][lang]
        return pool[lang_idx] if lang_idx < len(pool) else None

    build_component(
        component="public_domain",
        pools=pools,
        pick_safe=pick_safe,
        paraphrase=make_class3_paraphraser("public_domain"),
        output_path=args.output,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
