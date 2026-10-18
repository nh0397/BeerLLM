# BeerLLM

A reproducible, standard-library Python pipeline that fetches a bounded Brewtoad archive sample and generates source-grounded chat examples. This is a data feasibility pilot, not a trained model or a publication-ready dataset.

## Run

```sh
python3 scripts/curate.py --limit 100
```

Internet access is required. The maximum pilot limit is 1,000 files (GitHub's directory listing limit). The sample is filename-ordered and not representative of the archive. Runs overwrite curated outputs; raw files from previous runs may remain, but only files selected by the current run enter the outputs. The report and every record include the source commit.

## Outputs

- `data/raw/`: original XML files.
- `data/curated/recipes.candidate.jsonl`: parsed recipes, provenance, unit conflicts, missing process fields, and recipe-family splits.
- `data/curated/qa.candidate.jsonl`: chat examples about yeast and missing process information. Each includes its recipe context and split; this trains interpretation, not closed-book memorization.
- `data/curated/report.json`: counts, errors, and limitations.

Amounts explicitly labeled in display fields are normalized to kilograms. Conflicts with raw BeerXML amounts are flagged; no quantity questions are generated from these uncertain fields. Raw and display values remain available for review. Temperatures and durations remain source strings; no missing values are imputed. Recipe notes are preserved but are not treated as verified explanations.

Splits are assigned before QA generation using a hash of ingredient-name sets. Exact record duplicates are removed. Related recipes with matching names stay together; spelling variations and other near duplicates still require review before a real benchmark.

## Next stages

1. Confirm source reuse permissions before promoting candidates to a training release. Public availability is not a verified data license.
2. Audit a representative archive sample and inspect the Zenodo ingredient dataset. The current pilot fetches only Brewtoad, not AHA, Craft Beer & Brewing, or MongoDB.
3. Add licensed instructional sources with explicit attribution and original directions. Keep source instructions separate from generated wording and general brewing guidance.
4. Normalize ingredient aliases, validate numeric ranges and units, and cluster near duplicates across sources.
5. Add reviewed process and numerical QA examples. Build a human-reviewed test set independently of training templates.
6. Export separate train/validation/test chat JSONL files once quality and source rights are resolved, then compare a base model against QLoRA.

Never place MongoDB credentials in code or tracked files. A future MongoDB connector should read environment variables and use certificate verification.

The repository MIT license covers project code; it does not grant rights to third-party recipe data.
