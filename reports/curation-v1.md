# First collection curation pass

The collection reached its requested 1,500-record cap. After two exact-content duplicates, the input to curation contains **1,498 records**: 228 AHA, 415 BrewDog and 855 Brewer's Friend. Collection is stopped. The existing adapter and checkpoints are preserved; no new fine-tuning run was started for this pass.

## Audit results

| Review tier | Records | Meaning |
|---|---:|---|
| Instruction candidate | 50 | 49 AHA and one Brewer's Friend record mention all five stages and have no detected flags; substantive review still required |
| Needs enrichment | 1,407 | Missing steps/parameters, source warnings or other unresolved issues |
| Quarantined conflict | 16 | Contradictory pitching instructions or Fahrenheit/Celsius conversions |
| Excluded non-beer | 25 | Other drinks retained separately with their original source data |

The audit preserves 1,301 recipe families across the full input. The 50 conversation candidates split into 43 train, four validation and three test records. Related recipes share one partition; process variants are not discarded. This is a source-text compatibility dataset, not yet the conversational brewing-teacher dataset. **Approved training examples: zero.**

Common flags overlap: 491 records contain source equipment/calculation warnings, 431 have unresolved batch units, 418 have no written directions and 772 lack an explicit numeric temperature in their directions. Seventeen records contain a detected temperature conversion conflict; one is already in the excluded non-beer tier. These counts reflect what the parser detected, not a complete inventory of brewing errors. BrewDog's structured methods are preserved; they are not counted as a complete written procedure.

## Concrete findings

- A Kveiking We Will Go IPA gives incompatible pitching temperatures in different paragraphs. It remains quarantined instead of choosing one.
- Rogue Dead Guy Ale Clone pairs 40–50°F with 9–10°C. The conversion is inconsistent; source text is retained for review.
- More Fun Blonde Ale has yeast text embedded in a hop ingredient and repeated in a dedicated yeast section. It is flagged for a source-checked extraction repair.
- Bell's Two Hearted Ale Clone says to ferment warm without a numeric fermentation target. Its 64°F chilling instruction must not be silently treated as that missing fermentation target.
- Alt Ale Altbier's 2.5–2.7 volumes / 5–5.4 g/L refers to carbon dioxide, not a priming-sugar dose. A teaching answer must preserve that distinction.

For example, a useful beginner follow-up for Bell's is: “The boil is about 75 minutes, but the hops say 45 and 30 minutes. When do I add them?” A grounded draft answer would explain that these are minutes remaining: add the first 1.2 oz of Centennial after about 30 minutes of boiling and the second after about 45 minutes. The separate dry-hop addition comes later. This requires explanatory rewriting rather than copying a recipe JSON object. This example is a writing target, not a model output or an approved training row.

## Verification

All **35 unit tests passed**, including preserved beer styles, recovered yeast/other ingredients, fractional quantities, joined hop suffix provenance, conflict quarantine, family partitioning and refusal to overwrite a release with a different source snapshot. Python compilation and the runner's shell syntax check passed.

The cached SmolLM2 tokenizer and the existing assistant-only masking code processed every train/validation candidate:

| Partition | Used | Over context | Longest conversation | Supervised answer tokens |
|---|---:|---:|---:|---:|
| Train | 43 | 0 | 1,150 | 26,580 |
| Validation | 4 | 0 | 903 | 2,361 |

The context limit is 2,048 tokens. Test answers were not used for fitting or this trainer preflight. These measurements establish format compatibility only. They do not establish factual accuracy, expert approval, explanation quality or improved model behavior.

## Reproduce and continue

```bash
.venv/bin/python -m beerllm.curate
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --data-dir data/curated_v1/conversation_candidates --check-data
.venv/bin/python -m unittest discover -s tests
```

The source snapshot SHA-256 is `dd25318ae6f526941a4569f395799ccfa970078d74c959e17ae829b847f23881`. Local output is `data/curated_v1/`; the manifest also records the curation-code hash. Source records and generated datasets stay out of Git.

Next, source-check the strongest recipes and author the full beginner explanations and follow-ups, then check the rewritten examples and run a small learning experiment in a new output directory. Source procedure gaps and reuse rights remain unresolved; no expert review or brewing-accuracy claim has been made. See the [curation guide](../docs/CURATION.md) for exact rules and file meanings.
