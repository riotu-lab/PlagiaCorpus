# Human annotation

These are the sheets behind the *Human Annotation Reliability* analysis (manuscript, Table 8).

| File | Content |
|------|---------|
| `annotator_1.xlsx`, `annotator_2.xlsx` | The annotation sample: 300 queries, 50 per class (Class 1–5 and Safe), rated independently by each annotator. Items are `Sample_001` – `Sample_300`. |

The sample was drawn with `scripts/10_sample_for_annotation.py`. The Class 4 items come from the final (regenerated) Class 4 queries, balanced across the Wikipedia and public-domain components (`--classes "Class 4" --balance-domains`).

## Columns

| Column | Description |
|--------|-------------|
| `Sample_ID` | Item identifier, shared by both sheets. |
| `Hidden_True_Class` | Ground-truth class. It was hidden from annotators (blind review). |
| `Original_Text_ID` | `_id` of the passage shown in `Source_Text`. |
| `Query_Text` | The query to assess. |
| `Source_Text` | For Classes 1–5, the query's source passage. For Safe queries, an unrelated passage from the same component and language. |
| `Is_Plagiarism` | Rating (see below). |
| `Linguistic_Quality` | Rating (see below). |
| `Semantic_Preservation` | Rating (see below). |
| `Obfuscation_Difficulty` | Rating (see below). |
| `Annotator_Notes` | Free-text comments. |

## Rating guidelines

* **Is_Plagiarism** (Yes / No): whether the query is derived from the passage shown.
* **Linguistic_Quality** (1–5): grammar, readability and natural fluency of the query. This rating helps identify hallucinations, translation errors and algorithmic artifacts such as leetspeak.
  * 1 = incomprehensible or broken
  * 5 = flawless, native-like
* **Semantic_Preservation** (1–5): how accurately the core facts and meaning of the source are retained.
  * 1 = unrelated or completely different meaning
  * 5 = exact meaning preserved
* **Obfuscation_Difficulty** (1–3): how difficult it was to detect the reuse or link the query to its source.
  * 1 = obvious, direct copy
  * 2 = moderate, requires careful reading
  * 3 = highly deceptive, complex paraphrase

Notes on the completed sheets:
* Annotators also used the value 0, mostly for the Semantic_Preservation of Safe pairs and the Obfuscation_Difficulty of Class 1 direct copies.
* Some Obfuscation_Difficulty ratings go up to 5.
* `N/A` (or `NA`) marks a dimension that does not apply to the item.

## Agreement

```bash
python scripts/11_annotation_agreement.py
```

The script matches the two sheets on `Sample_ID`. For each dimension, it excludes items that either annotator left blank or marked `N/A`. It then reports N, raw agreement and Cohen's kappa. For the ordinal ratings, it also reports the quadratic weighted kappa and the share of ratings within one scale point. Results are written to `results/table8_annotation_agreement.csv`.
