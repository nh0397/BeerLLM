# First local training run

Training completed on October 18, 2026 (America/Los_Angeles). The local training workflow works, but the first adapter is **not yet a reliable assistant for complete brewing instructions**. Lower loss must not be confused with recipe correctness or task completion.

## Training

| Item | Result |
|---|---|
| Hardware | Intel MacBook Pro, 6-core Core i7, 32 GB RAM, CPU |
| Model | SmolLM2-360M-Instruct |
| Model revision | `a10cc1512eabd3dde888204e902eca88bddb4951` |
| Adapter | LoRA rank 8, query/value projections |
| Trainable parameters | 819,200 (about 0.226%) |
| Training examples | 364, from 182 recipes |
| Epochs / optimizer steps | 1 / 91 |
| Wall time | 3,395 seconds (56.6 minutes) |
| Mean training loss | 2.1233 |
| Final validation loss | 1.4423 on 8 validation examples |
| Overlength examples skipped | 0 |

Adapter: `outputs/beer-lora/adapter/`. Training metrics: `outputs/beer-lora/train_metrics.json`. Resumable checkpoints are preserved under `outputs/beer-lora/`.

## Evaluation

Three held-out recipe prompts were evaluated with the same supplied recipe context, using greedy decoding and an 800-token response limit. These are a small convenience sample (the first three test examples), not a representative brewing benchmark.

| Metric | Base model | LoRA adapter |
|---|---:|---:|
| Mean per-example assistant-token loss | 2.3880 | 1.5624 |
| Ingredient-name recall (mention proxy) | 0.6306 | 1.0000 |
| Brewing-stage mention coverage (keyword proxy) | 0.7333 | 0.7333 |
| Responses hitting token limit | 2/3 | 0/3 |
| Adapter answers delivering the requested ordered procedure | — | 0/3 |

The loss is measured against programmatic reference text, not expert-reviewed answers. Mention coverage is not factual accuracy: copying JSON can score highly without answering the question.

### Inspected answers

- **Hop Rocker (ID 5):** the base model invented ingredient quantities and repeated fermentation steps. The adapter copied the supplied recipe JSON rather than writing a brewing procedure.
- **Cult Lager (ID 13):** the base model confused hop/malt categories and proposed pitching yeast at 65°C. The adapter again returned recipe JSON without an ordered procedure.
- **Chaos Theory (ID 19):** the base model repeated transfers and invented durations. The adapter returned recipe-like JSON, omitted the supplied fermentation temperature, and did not explain how to brew it.

None of these model answers should be used as a brewing procedure. The adapter's lack of an ordered procedure was not caused by the response limit: all three adapter outputs stopped before reaching it. The small sample demonstrates a task failure, not a reliable estimate of performance across the catalog.

Full local answers, prompts, references and metrics are preserved in `outputs/evaluation.json`; the raw corpus and model outputs are not published to GitHub.

## Data limitations

The API catalog contains 415 recipes. Initial filtering retained 259 and set aside 156 with missing core fields, invalid quantities, or selected specialist processes. Of the retained recipes, 182 are training, 28 validation, and 49 test. Related names and matching ingredient sets were grouped before splitting, but this is not exhaustive near-duplicate detection.

Reference answers are programmatically composed guides, not expert-reviewed gold answers. Source ingredients, mash rests and fermentation temperatures are preserved. General process guidance is identified separately. Exact boil timing, some hop timing, conditioning and packaging details remain unresolved. None of these records should be represented as an expert-validated complete brew sheet.

## Run the model

From the repository root:

```bash
.venv/bin/python -m beerllm.chat --recipe-id 5 --prompt "Give me the ingredients and brewing procedure."
```

This explicitly includes the downloaded recipe in the prompt. To test model memory without supplying it, omit `--recipe-id`. Such recall is not established by the supplied-context evaluation. Use `--base` to compare the original model.

## Next experiment

Before increasing training time, curate complete instructional targets and manually review a small validation set. Test for actual ordered procedures and correct ingredient-quantity pairings; ingredient names and stage keywords alone reward copying JSON. Use validation data for iteration and reserve a new untouched test set for final claims after reacting to this test result. A larger base model or more training may help, but neither outcome is established by this run.
