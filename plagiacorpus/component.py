"""Generation of one source component (Wikipedia or public domain).

For every source passage, the component stores the passage itself and, in this order,
its Class 1 (direct copy), Class 2 (algorithmic obfuscation) and Class 3 (Qwen
paraphrase) queries, followed by one independently sampled Safe query.

Every query keeps an explicit `corpus_index` pointing to its source passage (None for
Safe queries), so source links never have to be inferred from record order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from tqdm.auto import tqdm

from .config import CLASS3_MODEL, GEN_ALGORITHMIC, GEN_DIRECT_COPY, GEN_HUMAN
from .io import load_json, save_json
from .prompts import CLASS3_SAMPLING, CLASS3_SYSTEM, CLASS3_USER
from .text import obfuscate

LANGUAGES = ("en", "ar")

# Provenance strings for the `source` field of each record type.
SOURCE_FORMATS = {
    "wikipedia": {
        "class1": "Wikipedia ({url})",
        "class2": "Algorithmically Modified from {url}",
        "class3": "Paraphrased by {model} from {url}",
        "safe": "Wikipedia ({url})",
    },
    "public_domain": {
        "class1": "{url}",
        "class2": "Obfuscated from {url}",
        "class3": "Paraphrased by {model} from {url}",
        "safe": "{url}",
    },
}

Paraphraser = Callable[[str, str], str]              # (text, lang) -> paraphrase
SafePicker = Callable[[int, int, str], dict | None]  # (queue idx, idx within lang, lang) -> passage


def load_or_build_pools(path: Path, build: Callable[[], dict]) -> dict:
    """Cache the collected passages so that an interrupted run resumes on identical inputs."""
    if path.exists():
        return load_json(path)
    pools = build()
    save_json(pools, path)
    return pools


def make_class3_paraphraser(component: str) -> Paraphraser:
    """Load Qwen2.5-14B-Instruct and return a (text, lang) -> paraphrase function."""
    from .llm import HFChatGenerator

    generator = HFChatGenerator(CLASS3_MODEL)
    sampling = CLASS3_SAMPLING[component]

    def paraphrase(text: str, lang: str) -> str:
        user = CLASS3_USER[component][lang].format(text=text)
        return generator.generate(CLASS3_SYSTEM[lang], user, **sampling)

    return paraphrase


def _query(text, qtype, label, source, generator, corpus_index):
    return {
        "text": text,
        "type": qtype,
        "label": label,
        "source": source,
        "generator_model": generator,
        "corpus_index": corpus_index,
    }


def build_component(
    component: str,
    pools: dict,
    pick_safe: SafePicker,
    paraphrase: Paraphraser,
    output_path: Path,
    save_every: int = 100,
) -> dict:
    """Generate the corpus and Class 1-3 + Safe queries of one component.

    `pools` = {"sources": {lang: [passage]}, "safe": {lang: [passage]}}, where a passage is
    {"text": ..., "url": ...}. English passages are processed first, then Arabic ones.
    Re-running with an existing `output_path` resumes after the last completed passage.
    """
    fmt = SOURCE_FORMATS[component]
    queue = [(passage, lang) for lang in LANGUAGES for passage in pools["sources"][lang]]

    if output_path.exists():
        state = load_json(output_path)
    else:
        state = {"component": component, "corpus": [], "queries": []}
    start = len(state["corpus"])
    seen_per_lang = Counter(doc["lang"] for doc in state["corpus"])

    for idx in tqdm(range(start, len(queue)), initial=start, total=len(queue), desc=component):
        passage, lang = queue[idx]
        lang_idx = seen_per_lang[lang]
        seen_per_lang[lang] += 1

        text, url, tag = passage["text"], passage["url"], lang.upper()
        state["corpus"].append({"text": text, "source": url, "lang": lang})
        state["queries"] += [
            _query(text, f"Class 1 ({tag})", 1, fmt["class1"].format(url=url), GEN_DIRECT_COPY, idx),
            _query(obfuscate(text, lang), f"Class 2 ({tag})", 1,
                   fmt["class2"].format(url=url), GEN_ALGORITHMIC, idx),
            _query(paraphrase(text, lang), f"Class 3 ({tag})", 1,
                   fmt["class3"].format(model=CLASS3_MODEL, url=url), CLASS3_MODEL, idx),
        ]
        safe = pick_safe(idx, lang_idx, lang)
        if safe is not None:
            state["queries"].append(
                _query(safe["text"], f"Safe ({tag})", 0, fmt["safe"].format(url=safe["url"]), GEN_HUMAN, None)
            )

        if (idx + 1) % save_every == 0:
            save_json(state, output_path)

    save_json(state, output_path)
    print(f"{component}: {len(state['corpus']):,} passages, {len(state['queries']):,} queries -> {output_path}")
    return state
