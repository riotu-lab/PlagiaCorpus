# PlagiaCorpus

Code for building and validating **PlagiaCorpus: A Bilingual, Adversarial, and Cross-Lingual Dataset for Forensic Plagiarism Detection** (submitted to *Scientific Data*).

PlagiaCorpus contains **243,529** Arabic and English text records: a source corpus of **38,737** reference passages and **204,792** queries (184,312 train / 20,480 test). The queries cover five text-reuse classes of increasing difficulty and a Safe (negative) class.

**Dataset:** `https://huggingface.co/datasets/<ORG>/PlagiaCorpus` <!-- TODO: replace with the final dataset URL -->

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

Python 3.10 or later is required. To validate and analyse the released files only, `pip install -e ".[analysis]"` is enough.

To regenerate the dataset you need GPUs. The released version was produced on a server with eight 80 GB A100 GPUs:

* **Class 3** runs `Qwen/Qwen2.5-14B-Instruct` through Hugging Face Transformers (bf16).
* **Classes 4 and 5** run `gemma4:31b` and `gemma4:26b` through [Ollama](https://ollama.com). Before steps 4 and 5, run `ollama pull gemma4:31b` and `ollama pull gemma4:26b`.

## Pipeline

| Step | Script | Manuscript section | Output |
|---|---|---|---|
| 1 | `01_build_wikipedia_component.py` | Source Corpus Construction; Classes 1–3; Negative Control Samples | `data/interim/wikipedia_component.json` |
| 2 | `02_build_public_domain_component.py` | same, public-domain component | `data/interim/public_domain_component.json` |
| 3 | `03_merge_components.py` | Dataset Integration and Assembly | `data/final/PlagiaCorpus_corpus.json` |
| 4 | `04_generate_class4_paraphrases.py` | LLM-Based Paraphrase Generation (Class 4) | `data/interim/class4/class4_worker*.json` |
| 5 | `05_generate_class5_crosslingual.py` | Cross-Lingual Text Reuse (Class 5) | `data/interim/queries_class5.json` |
| 6 | `06_assemble_and_split.py` | Dataset Integration and Assembly; Usage Notes | `data/final/PlagiaCorpus_queries_{train,test}.json` |
| 7 | `07_validate_dataset.py` | Technical Validation (structural integrity) | console report |
| 8 | `08_dataset_statistics.py` | Data Records (Table 2); Technical Validation (Table 7) | `results/table2_*.csv`, `results/table7_*.csv` |
| 9 | `09_language_distribution.py` | Data Records (Table 5) | `results/table5_*.csv` |
| 10 | `10_sample_for_annotation.py` | Human Annotation Reliability | `results/annotation_sample.xlsx` |
| 11 | `11_annotation_agreement.py` | Human Annotation Reliability (Table 8) | `results/table8_annotation_agreement.csv` |

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

# 6. Assemble all queries and create the stratified 90/10 split
python scripts/06_assemble_and_split.py

# 7-9. Technical validation and statistics
python scripts/07_validate_dataset.py
python scripts/08_dataset_statistics.py
python scripts/09_language_distribution.py

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

### Validating and analysing the released dataset

Download `PlagiaCorpus_corpus.json`, `PlagiaCorpus_queries_train.json` and `PlagiaCorpus_queries_test.json` into `data/final/`, then run steps 7–9. They take the same default paths. Step 11 reproduces Table 8 directly from the annotation sheets in [`annotation/`](annotation/README.md).

## Text-reuse classes

| Class | Reuse type | Generation | `type` values |
|---|---|---|---|
| Class 1 | Direct copy | Identity | `Class 1 (EN)`, `Class 1 (AR)` |
| Class 2 | Algorithmic modification | English leetspeak (e→3, a→@, i→1, o→0); Arabic tatweel (U+0640) inserted in words longer than two characters | `Class 2 (EN)`, `Class 2 (AR)` |
| Class 3 | Standard paraphrase | Qwen2.5-14B-Instruct, temperature 0.7 (top-p 0.9 for the PD component) | `Class 3 (EN)`, `Class 3 (AR)` |
| Class 4 | Complex paraphrase | gemma4:31b, temperature 0.7, top-p 0.9 | `Class 4 (EN)`, `Class 4 (AR)` |
| Class 5 | Cross-lingual reuse | gemma4:26b, translation with structural paraphrasing | `Class 5 (EN->AR)`, `Class 5 (AR->EN)` |
| Safe | Non-reused text | Independently sampled passage | `Safe (EN)`, `Safe (AR)` |

All prompts are in [`plagiacorpus/prompts.py`](plagiacorpus/prompts.py), exactly as used.

## Record format

**Corpus record** (`PlagiaCorpus_corpus.json`)

```json
{
    "_id": "WIKI_006573",
    "text": "a technical or scientific diploma or degree from a university ...",
    "domain": "Wikimedia Foundation (2024 Dumps)",
    "source": "https://en.wikipedia.org/wiki/..."
}
```

**Query record** (`PlagiaCorpus_queries_{train,test}.json`)

```json
{
    "text": "@ t3chn1c@l 0r sc13nt1f1c d1pl0m@ 0r d3gr33 fr0m @ un1v3rs1ty ...",
    "type": "Class 2 (EN)",
    "label": 1,
    "source": "Algorithmically Modified from https://en.wikipedia.org/wiki/...",
    "domain": "Wikimedia Foundation (2024 Dumps)",
    "generator_model": "None (Algorithmic Modification)",
    "original_text_id": "WIKI_006573"
}
```

| Field | In | Description |
|---|---|---|
| `_id` | corpus | Unique passage identifier (`WIKI_000000`, `PD_000000`, ...). |
| `text` | both | Passage or query text. |
| `domain` | both | Collection the record comes from. |
| `source` | both | Provenance (article URL, `HF://<dataset>/<title>`, and the generation step for queries). |
| `type` | queries | Reuse class and language, e.g. `Class 3 (AR)`. |
| `label` | queries | `1` = reused (Classes 1–5), `0` = Safe. |
| `generator_model` | queries | Model that produced the query, `None (...)` for rule-based classes, `Human (Original Text)` for Safe. |
| `original_text_id` | queries | `_id` of the source passage; `UNKNOWN_ID` for Safe queries. |

## Reproducibility notes

* LLM generation uses stochastic decoding without fixed seeds, and the Wikipedia crawl reflects the encyclopedia at crawl time. Regenerating the data therefore gives a dataset that is comparable to the released one but not identical. The released files are the reference version. Steps 7–11 reproduce the reported statistics from them.
* Sampling of Class 5 sources, the train/test split and the annotation sample all use seed 42.
* `07_validate_dataset.py` checks every query against the passage referenced by its `original_text_id`. For example, it checks that Class 1 equals the source, that Class 2 equals the obfuscated source, and that Class 5 languages match the translation direction.

## License

The code is released under the [MIT License](LICENSE). The dataset is distributed under the license stated on its dataset card. Source passages keep the terms of their original collections; for example, Wikipedia text is available under CC BY-SA 4.0.

## Acknowledgments

This project was funded by the Saudi Ministry of Commerce. The authors acknowledge the support of Prince Sultan University.

## Citation

Citation details will be added upon publication.
