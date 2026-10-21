"""Step 4 - Class 4 complex paraphrases with gemma4:31b (Ollama), parallelisable across GPUs.

Every passage of the merged corpus is paraphrased once in its own language. The corpus
is partitioned round-robin: worker k handles passages k, k+N, k+2N, ... and writes its
own file `class4_worker{k}.json`, so workers never contend for the same output. Failed
generations are not written and are retried automatically on the next run.

Single GPU:
    python scripts/04_generate_class4_paraphrases.py

Eight GPUs (one Ollama server per GPU, ports 11434-11441):
    for k in $(seq 0 7); do
        python scripts/04_generate_class4_paraphrases.py \
            --worker-id $k --num-workers 8 --ollama-url http://localhost:$((11434 + k)) &
    done; wait
"""

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from plagiacorpus.config import CLASS4_DIR, CLASS4_MODEL, CORPUS_FILE
from plagiacorpus.io import load_json, save_json
from plagiacorpus.llm import OllamaClient
from plagiacorpus.prompts import CLASS4_PROMPT
from plagiacorpus.text import language_of


def class4_query(doc: dict, paraphrase: str, model: str) -> dict:
    lang = language_of(doc["text"])
    return {
        "text": paraphrase,
        "type": f"Class 4 ({lang.upper()})",
        "label": 1,
        "source": f"Paraphrased by {model} from {doc.get('source') or doc['_id']}",
        "domain": doc["domain"],
        "generator_model": model,
        "original_text_id": doc["_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--out-dir", type=Path, default=CLASS4_DIR)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default=CLASS4_MODEL)
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per request.")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many passages (for testing).")
    args = parser.parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        parser.error("--worker-id must be in [0, --num-workers)")

    client = OllamaClient(args.model, args.ollama_url, timeout=args.timeout)
    client.check_available()

    output_path = args.out_dir / f"class4_worker{args.worker_id}.json"
    queries = load_json(output_path) if output_path.exists() else []
    done = {q["original_text_id"] for q in queries}

    # Assign the slice from the full corpus first, so the partition is identical on every resume.
    assigned = load_json(args.corpus)[args.worker_id::args.num_workers]
    todo = [doc for doc in assigned if doc["_id"] not in done][: args.limit]
    print(f"[worker {args.worker_id}] assigned {len(assigned):,} | done {len(done):,} | to process {len(todo):,}")

    failures = 0
    for i, doc in enumerate(tqdm(todo, desc=f"Class 4 (worker {args.worker_id})"), start=1):
        prompt = CLASS4_PROMPT[language_of(doc["text"])].format(text=doc["text"])
        paraphrase = client.generate(prompt)
        if paraphrase is None:
            failures += 1
        else:
            queries.append(class4_query(doc, paraphrase, args.model))
        if i % args.save_every == 0:
            save_json(queries, output_path)
    save_json(queries, output_path)

    print(f"[worker {args.worker_id}] generated {len(todo) - failures:,}, failed {failures:,} -> {output_path}")
    if failures:
        print("Re-run the same command to retry the failed passages.")


if __name__ == "__main__":
    main()
