# PlagiaCorpus

Code for building and validating **PlagiaCorpus: A Bilingual, Adversarial, and Cross-Lingual Dataset for Forensic Plagiarism Detection** (submitted to *Scientific Data*).

PlagiaCorpus contains **198,643** Arabic and English text records: a source corpus of **34,885** reference passages and **163,758** queries (**147,381** train / **16,377** test). The queries cover five text-reuse classes of increasing difficulty and a Safe (negative) class. The train/test split is grouped by source passage: no passage contributes queries to both splits.

**Dataset:** [PlagiaCorpus on Hugging Face](https://huggingface.co/datasets/riotu-lab/PlagiaCorpus) · DOI [10.57967/hf/11369](https://doi.org/10.57967/hf/11369)

## Repository structure

```
PlagiaCorpus/
├── plagiacorpus/                 # shared library
│   ├── config.py                 #   model names, labels, domains, topics, file paths
│   ├── prompts.py                #   verbatim LLM prompts (Figures 2-4)
│   ├── text.py                   #   language heuristic, word count, Class 2 obfuscation
│   ├── component.py              #   corpus + Class 1-3 + Safe generation for one component
│   ├── llm.py                    #   Transformers and Ollama clients
│   └── io.py                     #   JSON helpers (atomic writes)
├── scripts/                      # numbered pipeline steps (see below)
├── notebooks/
│   └── PlagiaCorpus_curation.ipynb   # curation and grouped train/test split of the release
├── annotation/                   # annotated sample (300 queries, two annotators)
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
git clone https://github.com/riotu-lab/PlagiaCorpus.git
cd PlagiaCorpus
pip install -e ".[all]"
```

Python 3.10 or later is required. To analyse the released files only, `pip install -e ".[analysis]"` is enough. The curation notebook additionally uses `lingua-language-detector`, `sentence-transformers` and `pyarrow`; it installs them itself and runs on Google Colab, with a GPU recommended.

To regenerate the dataset you need GPUs. The released version was produced on a server with eight 80 GB A100 GPUs:

* **Class 3** runs `Qwen/Qwen2.5-14B-Instruct` through Hugging Face Transformers (bf16, at most 256 new tokens).
* **Classes 4 and 5** run `gemma4:31b` and `gemma4:26b` through [Ollama](https://ollama.com). Before steps 4 and 5, run `ollama pull gemma4:31b` and `ollama pull gemma4:26b`.

## Pipeline

Steps 1–6 generate the initial assembly. The curation notebook then produces the released version.

| Step | Script | Manuscript section | Output |
|---|---|---|---|
| 1 | `01_build_wikipedia_component.py` | Source Corpus Construction; Classes 1–3; Negative Control Samples | `data/interim/wikipedia_component.json` |
| 2 | `02_build_public_domain_component.py` | same, book component | `data/interim/public_domain_component.json` |
| 3 | `03_merge_components.py` | Dataset Integration and Assembly | `data/final/PlagiaCorpus_corpus.json` |
| 4 | `04_generate_class4_paraphrases.py` | LLM-Based Paraphrase Generation (Class 4) | `data/interim/class4/class4_worker*.json` |
| 5 | `05_generate_class5_crosslingual.py` | Cross-Lingual Text Reuse (Class 5) | `data/interim/queries_class5.json` |
| 6 | `06_assemble_and_split.py` | Dataset Integration and Assembly | `data/final/PlagiaCorpus_queries_{train,test}.json` (initial assembly) |
| C | `notebooks/PlagiaCorpus_curation.ipynb` | Curation and Train/Test Split; Data Records; Technical Validation (Tables 2, 5, 7) | released files + `reports/` |
| 10 | `10_sample_for_annotation.py` | Human Annotation Reliability | `results/annotation_sample.xlsx` |
| 11 | `11_annotation_agreement.py` | Human Annotation Reliability (Table 8) | `results/table8_annotation_agreement.csv` |

Steps 7–9 (`07_validate_dataset.py`, `08_dataset_statistics.py`, `09_language_distribution.py`) validate and describe the initial assembly produced by step 6. The statistics of the released version are computed by the curation notebook.

Run every step from the repository root:

```bash
# 1-2. Source components: passages + Class 1, 2, 3 and Safe queries
python scripts/01_build_wikipedia_component.py
python scripts/02_build_public_domain_component.py

# 3. Merge into the bilingual source corpus (assigns WIKI_xxxxxx / PD_xxxxxx ids)
python scripts/03_merge_components.py

# 4. Class 4 complex paraphrases (gemma4:31b)
python scripts/04_generate_class4_paraphrases.py

# 5. Class 5 cross-lingual reuse (gemma4:26b)
python scripts/05_generate_class5_crosslingual.py --per-direction 2000

# 6. Assemble all queries (initial assembly)
python scripts/06_assemble_and_split.py

# C. Curation and grouped split: open notebooks/PlagiaCorpus_curation.ipynb,
#    set IN_DIR to the folder holding the three files from steps 3 and 6, and run all cells

# 10-11. Human annotation (sampling, then agreement on the sheets in annotation/)
python scripts/10_sample_for_annotation.py
python scripts/11_annotation_agreement.py
```

The generation steps (1, 2, 4 and 5) save their progress regularly. If one is interrupted, re-run it and it resumes where it stopped. A failed LLM call is never replaced by the input text; it is retried on the next run.

**Multi-GPU Class 4 generation.** Start one Ollama server per GPU, then one worker per server:

```bash
for k in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$k OLLAMA_HOST=127.0.0.1:$((11434 + k)) ollama serve &
done
for k in $(seq 0 7); do
    python scripts/04_generate_class4_paraphrases.py \
        --worker-id $k --num-workers 8 --ollama-url http://127.0.0.1:$((11434 + k)) &
done
wait
```

## Curation

The curation notebook turns the initial assembly into the released version. Every step is deterministic (seed 42), and every removed record is logged in `reports/removed_records.csv`.

1. **Collection labels and attribution.** Each record gets the name of its source collection: Wikipedia (accessed April 2026), The Arabic E-Book Corpus (Hindawi Foundation), or PleIAs English-PD. Each record also gets the licence of its text and a source citation that includes the book title.
2. **Language filtering.** Arabic-script passages are accepted as Arabic. A Latin-script passage is treated as English unless two signals agree that it is not: fewer than 5% English function words, and a confidence below 0.2 for English according to [Lingua](https://github.com/pemistahl/lingua-py). Passages that are not English or Arabic are removed with all queries derived from them.
3. **LLM output cleaning (Classes 3–5 only).** Introductory phrases, trailing notes and markdown are stripped. Refusals, echoed prompts, outputs in the wrong language or script, outputs containing characters of an unrelated script, and outputs identical to their source are removed. Outputs that may be truncated, or that are much shorter than their source, are kept and flagged in `quality_flags`.
4. **De-duplication.** Safe queries identical to a source passage or to another Safe query are removed. Source and Safe passages are embedded with [LaBSE](https://huggingface.co/sentence-transformers/LaBSE): Safe passages with a cosine similarity of at least 0.85 to a source passage are removed, and near-duplicate source passages are grouped.
5. **Safe obfuscation.** 25% of the Safe queries in each language receive the same character-level rules as Class 2.
6. **Grouped split.** Each source passage, merged with its exact and near duplicates, forms a group. All queries of a group go to the same split. Groups are assigned to the test split within strata (collection × language; Safe queries separately) until 10% of each stratum's queries is reached. The notebook verifies that no source passage, and no identical query text, occurs in both splits.

| Records removed | Count |
|---|---:|
| Source passages not in English/Arabic or inconsistent with their queries | 3,852 |
| Queries derived from a removed source passage | 15,951 |
| Safe queries in a language other than their label | 5,920 |
| Safe queries duplicating another Safe query | 14,705 |
| Safe queries identical to a source passage | 2,330 |
| Safe queries near-duplicate of a source passage (LaBSE) | 68 |
| LLM outputs with characters of an unrelated script | 1,756 |
| LLM outputs in the wrong language or script | 284 |
| LLM refusals | 20 |

## Text-reuse classes

| Class | Reuse type | Generation | `type` values | Train | Test |
|---|---|---|---|---:|---:|
| Class 1 | Direct copy | Identity | `Class 1 (EN)`, `Class 1 (AR)` | 31,400 | 3,485 |
| Class 2 | Algorithmic modification | English leetspeak (e→3, a→@, i→1, o→0); Arabic tatweel (U+0640) inserted after the first character of words longer than two characters | `Class 2 (EN)`, `Class 2 (AR)` | 31,400 | 3,485 |
| Class 3 | Standard paraphrase | Qwen2.5-14B-Instruct, temperature 0.7 (top-p 0.9 for the book component), 256 new tokens | `Class 3 (EN)`, `Class 3 (AR)` | 29,618 | 3,313 |
| Class 4 | Complex paraphrase | gemma4:31b, temperature 0.7, top-p 0.9 | `Class 4 (EN)`, `Class 4 (AR)` | 29,270 | 3,224 |
| Class 5 | Cross-lingual reuse | gemma4:26b, translation with structural paraphrasing | `Class 5 (EN->AR)`, `Class 5 (AR->EN)` | 11,551 | 1,298 |
| Safe | Non-reused text | Independently sampled passage; 25% with Class 2 obfuscation | `Safe (EN)`, `Safe (AR)` | 14,142 | 1,572 |
| **Total** | | | | **147,381** | **16,377** |

All prompts are in [`plagiacorpus/prompts.py`](plagiacorpus/prompts.py), exactly as used.

## Record format

Each file is provided in JSON and Parquet.

**Corpus record** (`PlagiaCorpus_corpus`)

```json
{
    "_id": "WIKI_006573",
    "text": "a technical or scientific diploma or degree from a university ...",
    "language": "en",
    "domain": "Wikipedia (Wikimedia Foundation), accessed April 2026",
    "source": "Wikipedia (https://en.wikipedia.org/wiki/...)",
    "source_license": "CC BY-SA 4.0"
}
```

**Query record** (`PlagiaCorpus_queries_{train,test}`)

```json
{
    "_id": "Q_000123",
    "text": "@ t3chn1c@l 0r sc13nt1f1c d1pl0m@ 0r d3gr33 fr0m @ un1v3rs1ty ...",
    "type": "Class 2 (EN)",
    "reuse_class": "Class 2",
    "language": "en",
    "source_language": "en",
    "transformation": "algorithmic",
    "label": 1,
    "source": "Algorithmically Modified from https://en.wikipedia.org/wiki/...",
    "domain": "Wikipedia (Wikimedia Foundation), accessed April 2026",
    "source_license": "CC BY-SA 4.0",
    "generator_model": "None (Algorithmic Modification)",
    "original_text_id": "WIKI_006573",
    "quality_flags": []
}
```

| Field | In | Description |
|---|---|---|
| `_id` | both | Unique identifier: `WIKI_000000` / `PD_000000` for source passages, `Q_000000` for queries. |
| `text` | both | Passage or query text. |
| `language` | both | Language of the text (`en` or `ar`). |
| `domain` | both | Source collection: `Wikipedia (Wikimedia Foundation), accessed April 2026`, `The Arabic E-Book Corpus (Hindawi Foundation)` or `PleIAs English-PD`. |
| `source` | both | Attribution: the Wikipedia article URL or a citation of the book collection with the book title; for queries, preceded by the transformation applied. |
| `source_license` | both | Licence of the underlying text: `CC BY-SA 4.0`, `CC BY 4.0` or `Public domain`. |
| `type` | queries | Reuse class and language, e.g. `Class 3 (AR)`, `Class 5 (EN->AR)`. |
| `reuse_class` | queries | `Class 1` to `Class 5`, or `Safe`. |
| `source_language` | queries | Language of the source passage (`null` for Safe queries). |
| `transformation` | queries | `verbatim`, `algorithmic`, `llm_paraphrase`, `llm_cross_lingual` or `none`. |
| `label` | queries | `1` = reused (Classes 1–5), `0` = Safe. |
| `generator_model` | queries | Model that produced the query, `None (...)` for rule-based classes, `Human (Original Text)` for Safe. |
| `original_text_id` | queries | `_id` of the source passage; `null` for Safe queries. |
| `quality_flags` | queries | Automatic flags for LLM outputs: `possibly_truncated`, `much_shorter_than_source`. |

## Reproducibility notes

* LLM generation uses stochastic decoding without fixed seeds, and the Wikipedia crawl reflects the encyclopedia at the time of the crawl (April 2026). Regenerating the data therefore gives a dataset that is comparable to the released one but not identical. The released files are the reference version.
* Sampling of Class 5 sources, the curation steps, the train/test split and the annotation sample all use seed 42.
* The human annotation (Table 8) was carried out on a sample drawn before curation; `11_annotation_agreement.py` reproduces it from the sheets in [`annotation/`](annotation/README.md).

## License

The code is released under the [MIT License](LICENSE). The dataset is released under **CC BY-SA 4.0**, as required by the Wikipedia ShareAlike terms. Passages from The Arabic E-Book Corpus ([Hallberg 2025](https://doi.org/10.5878/7rbh-gy93)) are redistributed under the attribution terms of its CC BY 4.0 license, and passages from PleIAs English-PD are in the public domain. The `source` and `source_license` fields give the attribution of every record.

## Acknowledgments

This research was conducted with the support of the Cultural Research Grants Program launched by the Ministry of Culture in the Kingdom of Saudi Arabia. The authors acknowledge the support of Prince Sultan University.

## Citation

If you use PlagiaCorpus, please cite the dataset:

```bibtex
@misc{adel_ammar_2026,
  author    = {Adel Ammar and Serry Sibaee and Wadii Boulila},
  title     = {PlagiaCorpus (Revision c603a43)},
  year      = {2026},
  url       = {https://huggingface.co/datasets/riotu-lab/PlagiaCorpus},
  doi       = {10.57967/hf/11369},
  publisher = {Hugging Face}
}
```

The citation of the accompanying article will be added upon publication.
