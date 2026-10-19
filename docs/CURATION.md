# Curate the collected recipes

Collection stopped at the requested 1,500 saved records. Conservative exact deduplication left 1,498 records. This command works entirely from that local export; it does not download pages or start training.

```bash
.venv/bin/python -m beerllm.curate
cat data/curated_v1/manifest.json

# Check the candidate conversation format with the cached model tokenizer.
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --data-dir data/curated_v1/conversation_candidates --check-data
```

Use `--input path/to/recipes.jsonl --output data/curated_v2` for another source snapshot. A release directory refuses a different input hash. Rerunning with the same input rebuilds its derived audits and conversations; the manifest records the curation code hash as well. This freezes the input, not an approved training release. Use a new output directory when preserving an older audit version matters.

## Files

| Local path under `data/curated_v1/` | Purpose |
|---|---|
| `source_snapshot.jsonl` | Exact copy of the input bytes, with SHA-256 in the manifest |
| `recipes.jsonl` | All records, source references, original process text, conversions, repair logs, review flags, families and splits |
| `instruction_candidate.jsonl` | Ingredients, a resolved batch volume, stage mentions and no detected flags; requires substantive review |
| `needs_enrichment.jsonl` | Missing parameters, incomplete instructions or other unresolved flags |
| `quarantine_conflict.jsonl` | Detected contradictory temperatures; source values retained unchanged |
| `excluded_nonbeer.jsonl` | Other drinks set aside without deleting their source records |
| `conversation_candidates/{train,validation,test}.jsonl` | Source-text conversation drafts with `candidate_not_approved` status |
| `manifest.json` | Source/code hashes, counts, flags and candidate partition sizes |

Each JSONL line is one object. The conversation files have `messages` containing `system`, `user` and `assistant` messages; source and review metadata sit outside the messages. The existing trainer can read this format, but it does not enforce review status. Do not point a long training run at these drafts until the required review and teaching rewrite are complete. The default runner still uses the older dataset.

## Cleaning and grouping rules

The original input remains intact. Cleaning normalizes whitespace and fraction notation, joins separated hop alpha-acid suffixes, removes ingredient section headings and recovers Brewer's Friend yeast and other ingredients from its public text representation. It preserves hop additions separately, even when they use the same variety.

Metric batch volumes are preferred. Explicit US gallons can be converted; unqualified gallons remain unresolved. Supplementary leading mass conversions retain the original amount and unit. An ounce is interpreted as weight in this supplementary field, so liquid ingredients need review. Answers keep original quantities rather than substituting these conversions.

The audit flags source equipment/calculation warnings, selected extract/mash inconsistencies, embedded yeast extraction artifacts and missing numeric temperatures. Fahrenheit/Celsius pairs differing by more than 1.5°C are quarantined. A manually observed contradictory pitching instruction is also quarantined. These checks are deliberately incomplete: they cannot establish brewing correctness, detect every contradiction or replace a brewer's judgment.

Normalized matching names, identical cleaned content or identical quantity-bearing ingredient lists connect recipe families. Family members share a deterministic 80/10/10 train/validation/test allocation. Process variants remain distinct records. Family matching is conservative and language/unit dependent; unrecognized aliases can remain. The earlier 44 similar-title review groups remain under `data/collection/`.

## Review before fine-tuning

For each candidate, check the original page snapshot against ingredients, batch size, mash/steep, boil and additions, cooling/pitching, fermentation and packaging. A stage keyword can occur in a warning or an incomplete sentence. A single temperature somewhere in the directions does not establish the fermentation target. Confirm missing parameters explicitly instead of completing them from a generic template.

Then author a beginner-friendly full recipe and useful follow-up conversations: what a step means, why it matters, which mistakes to avoid and how to check completion. Label general brewing explanations separately from source recipe facts and support them with suitable references. Keep every task from a recipe family in its assigned partition. Check tokens again after rewriting; these drafts' lengths do not predict final teaching-answer lengths.

The current dataset has **zero approved training examples**. A tokenizer check proves input compatibility, not answer quality. See the measured [curation report](../reports/curation-v1.md).
