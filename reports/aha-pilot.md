# AHA ingestion and fine-tuning-format pilot

Verified October 18, 2026. This pilot checks whether public recipe pages can supply actual instructions in the format accepted by the existing LoRA trainer. It does not establish that the model has learned to teach brewing.

## Actual results

- Five selected public pages imported successfully, without membership credentials: Your First SMaSH Ale, A New Hope Pale Ale, Nearly Nirvana Pale Ale, Deck-Head Stout, and Bru’n Dry Stout.
- Ingredients, source specifications and original written directions were extracted. No recipe steps were generated to fill gaps.
- Five conversation candidates exported: three training, one validation and one test. Their ordinary user prompts ask for recipes; no raw JSON is included in those prompts.
- Every complete example fits the 2,048-token context. Token counts across the pilot are 406, 476, 498, 691 and 896. Decoding the supervised tokens confirmed the entire assistant answer is preserved.
- An actual CPU LoRA compatibility run completed one optimizer step over two training examples, evaluated on one validation example, and saved an adapter in `outputs/aha-format-smoke/adapter/`. Runtime was about 16 seconds. This adapter is only a technical test, not a useful fine-tuned model.
- Seventeen unit tests pass. They cover parsing, missing/restricted pages, original units/order, URL scope, natural conversation format, grouped splits and training masks, alongside prior pipeline tests.

## Instruction gaps found

| Recipe | Stage mentions absent from extracted directions |
|---|---|
| Your First SMaSH Ale | None |
| A New Hope Pale Ale | Packaging |
| Nearly Nirvana Pale Ale | Cooling |
| Deck-Head Stout | Boil, cooling, packaging |
| Bru’n Dry Stout | Cooling, fermentation, packaging |

This is a keyword audit, not a completeness verdict. For example, a stage might be present under other wording, or mentioned without enough detail. Even the recipe with no missing stage mentions requires review of fermentation completion criteria and recipe-specific quantities and timings. A source can contain real instructions while still assuming an experienced brewer.

## Decision

The pipeline works for collection, extraction and fine-tuning input compatibility. The next step is substantive recipe review and source-supported teaching conversations: clear actions, explanations, mistakes to avoid and appropriate clarification. Scaling collection or running another full epoch before this work would repeat the original data-quality problem.

The five-page pilot is too small for meaningful model-quality or generalization claims. Candidate outputs remain local under `data/aha/`; source reuse terms remain unverified. The old PunkAPI dataset and model are preserved.

## Reproduce

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m beerllm.aha --urls-file configs/aha-sample.txt
.venv/bin/python -m beerllm.aha_dataset
.venv/bin/python -m beerllm.train --data-dir data/aha/training_candidates --check-data
```

Use a fresh output directory to repeat the real optimizer check:

```bash
.venv/bin/python -m beerllm.train --data-dir data/aha/training_candidates \
  --output outputs/aha-smoke-repeat --max-steps 1 --limit 2 --eval-limit 1
```
