"""Step 1 - Wikipedia component.

Crawls English and Arabic Wikipedia paragraphs (> 200 characters, section headings
removed) for a fixed list of topics, collects independent Safe control pools, and
generates Class 1, Class 2, Class 3 (Qwen2.5-14B-Instruct) and Safe queries.

Usage:
    python scripts/01_build_wikipedia_component.py
"""

import argparse
import time
import warnings
from pathlib import Path

from tqdm.auto import tqdm

from plagiacorpus.component import build_component, load_or_build_pools, make_class3_paraphraser
from plagiacorpus.config import WIKI_COMPONENT_FILE, WIKI_SAFE_TOPICS, WIKI_TOPICS

MIN_PARAGRAPH_CHARS = 200
RESULTS_PER_TOPIC = 20
REQUEST_DELAY_S = 0.2


def crawl_wikipedia(topics: list[str], lang: str, target: int) -> list[dict]:
    """Collect up to `target` paragraphs from the pages returned for each topic search."""
    import wikipedia

    warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")
    wikipedia.set_lang(lang)
    passages: list[dict] = []

    with tqdm(total=target, desc=f"Crawling {lang.upper()} Wikipedia") as bar:
        for topic in topics:
            if len(passages) >= target:
                break
            try:
                titles = wikipedia.search(topic, results=RESULTS_PER_TOPIC)
            except Exception:
                continue
            for title in titles:
                if len(passages) >= target:
                    break
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                except Exception:
                    continue
                for paragraph in page.content.split("\n\n"):
                    paragraph = paragraph.strip()
                    if len(paragraph) > MIN_PARAGRAPH_CHARS and "==" not in paragraph:
                        passages.append({"text": paragraph, "url": page.url})
                        bar.update(1)
                        if len(passages) >= target:
                            break
                time.sleep(REQUEST_DELAY_S)
    return passages


def collect_pools(max_per_lang: int) -> dict:
    sources = {lang: crawl_wikipedia(WIKI_TOPICS[lang], lang, max_per_lang) for lang in ("en", "ar")}
    safe = {lang: crawl_wikipedia(WIKI_SAFE_TOPICS[lang], lang, len(sources[lang])) for lang in ("en", "ar")}
    return {"sources": sources, "safe": safe}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=WIKI_COMPONENT_FILE)
    parser.add_argument("--max-paragraphs-per-lang", type=int, default=300_000,
                        help="Upper bound per language; in practice the topic list is exhausted first.")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    pools_file = args.output.with_name(args.output.stem + "_pools.json")
    pools = load_or_build_pools(pools_file, lambda: collect_pools(args.max_paragraphs_per_lang))

    def pick_safe(idx: int, _lang_idx: int, lang: str):
        pool = pools["safe"][lang]
        return pool[idx % len(pool)] if pool else None

    build_component(
        component="wikipedia",
        pools=pools,
        pick_safe=pick_safe,
        paraphrase=make_class3_paraphraser("wikipedia"),
        output_path=args.output,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
