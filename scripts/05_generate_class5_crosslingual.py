"""Step 5 - Class 5 cross-lingual reuse with gemma4:26b (Ollama).

The merged corpus is split by language and source passages are sampled for both
directions (EN->AR and AR->EN). Each sampled passage is translated with light
structural paraphrasing. Passages that already have a Class 5 query are excluded from
sampling, so re-running the script tops up each direction to `--per-direction`
without reusing a source.

Usage:
    python scripts/05_generate_class5_crosslingual.py --per-direction 2000
"""

import argparse
import random
from pathlib import Path

from tqdm.auto import tqdm

from plagiacorpus.config import CLASS5_FILE, CLASS5_MODEL, CORPUS_FILE
from plagiacorpus.io import load_json, save_json
from plagiacorpus.llm import OllamaClient
from plagiacorpus.prompts import CLASS5_PROMPT
from plagiacorpus.text import language_of

DIRECTIONS = {"EN->AR": "en", "AR->EN": "ar"}  # direction -> source language
MIN_OUTPUT_CHARS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--output", type=Path, default=CLASS5_FILE)
    parser.add_argument("--per-direction", type=int, default=2000, help="Target number of queries per direction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default=CLASS5_MODEL)
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per request.")
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    client = OllamaClient(args.model, args.ollama_url, timeout=args.timeout)
    client.check_available()

    queries = load_json(args.output) if args.output.exists() else []
    used = {q["original_text_id"] for q in queries}
    corpus = [doc for doc in load_json(args.corpus) if doc["_id"] not in used]
    rng = random.Random(args.seed)

    for direction, source_lang in DIRECTIONS.items():
        existing = sum(q["type"] == f"Class 5 ({direction})" for q in queries)
        pool = [doc for doc in corpus if language_of(doc["text"]) == source_lang]
        sample = rng.sample(pool, min(max(0, args.per_direction - existing), len(pool)))
        print(f"{direction}: {existing:,} existing, {len(sample):,} to generate ({len(pool):,} candidates)")

        for i, doc in enumerate(tqdm(sample, desc=f"Class 5 {direction}"), start=1):
            translation = client.generate(CLASS5_PROMPT[direction].format(text=doc["text"]))
            if translation and len(translation) > MIN_OUTPUT_CHARS:
                queries.append({
                    "text": translation,
                    "type": f"Class 5 ({direction})",
                    "label": 1,
                    "source": f"Translated by {args.model} from {doc.get('source') or doc['_id']}",
                    "domain": doc["domain"],
                    "generator_model": args.model,
                    "original_text_id": doc["_id"],
                })
            if i % args.save_every == 0:
                save_json(queries, args.output)
        save_json(queries, args.output)

    print(f"Class 5 queries: {len(queries):,} -> {args.output}")


if __name__ == "__main__":
    main()
