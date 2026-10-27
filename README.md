# BeerLLM

A local LoRA fine-tuning experiment for beer recipes and brewing procedures. Runs on an Intel Mac with 32 GB RAM. The latest experiment uses Qwen2.5-1.5B-Instruct on CPU; the earlier SmolLM2-360M runs are preserved. No cloud account or paid API is needed.

**One-recipe demo:** Helicious Helles now passes all twelve fresh source checks: six standalone questions and a six-turn conversation. The matched base passes 0/12 under the same greedy policy and response budgets. This is one trained recipe, with assistant source inspection rather than expert brewing review. See the [actual conversation](reports/helles-conversation.md), [measured before/after report](reports/single-recipe-run.md) and [commands](docs/SINGLE-RECIPE.md).

**Previous broader experiment:** a refined Qwen adapter has finished training on 197 distinct conversations from eleven source-checked recipes. Short factual rows receive explicit replay weighting; they are not additional independent examples. See the [refined run report](reports/refinement-run.md) for measured progress and the [reproduction guide](docs/REFINEMENT.md) for commands. The epoch-two checkpoint is selected by lowest validation loss. Strict development passes improve from **1/8 to 5/8**, but complete guides still invent steps and fail. Both reserved full-guide cases also fail before and after (0/2). The model is testable, but full recipe generation remains unreliable.

**Why the refinement:** the [earlier instructional model](reports/instruction-run.md) and [first Qwen trial](reports/instruction-v2-run.md) failed strict answer review despite lower reference perplexity. A varied two-fact fitting check now passes both exact questions and both fixed rewordings, but that does not establish full brewing competence. Every broad run retains actual generated answers and a separate factual review. References are not expert brewer-reviewed.

## Multi-source recipe collection

Collect AHA and Brewer's Friend public pages alongside the BrewDog API snapshot, with resumable downloads and conservative exact deduplication. See the [collection guide](docs/COLLECTION.md) for full-queue commands, progress checks and stopping/resuming. Full collection runs for days; it is separate from fine-tuning.

**Current data:** collection stopped at the requested 1,500 saved records, leaving 1,498 after exact deduplication. The [first curation pass](reports/curation-v1.md) separates 50 instruction candidates, 1,407 records needing enrichment, 16 instruction conflicts and 25 non-beer records. The previous broader teaching experiment uses eleven training recipes, four validation families and two separately reserved final families. It does not train on all 1,498 records. Run `.venv/bin/python -m beerllm.curate` to rebuild the local audit. See the [curation guide](docs/CURATION.md).

**Teaching data:** recipes become ingredient lists, ordered beginner guides, direct questions, conversational follow-ups and questions with actual source notes. Source text, hashes, family splits and missing details are retained. The [metric guide](docs/METRICS.md) explains loss, perplexity and strict answer review; lower loss alone cannot approve the model.

## AHA instruction-data pipeline

This importer preserves actual written recipe directions from selected public AHA pages. See the [roadmap](docs/ROADMAP.md) for milestones and acceptance checks.

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m beerllm.aha --urls-file configs/aha-sample.txt
.venv/bin/python -m beerllm.aha_dataset
.venv/bin/python -m beerllm.train --data-dir data/aha/training_candidates --check-data
```

This extracts source instructions, exports natural-language conversation candidates, and checks tokenization without training. Data is saved separately under `data/aha/`. Review the candidate answers before using them in a long run. The existing one-command runner below still reproduces the older PunkAPI experiment; it does not automatically switch to AHA data.

## Reproduce the first API-data experiment

Python 3.12 is tested. On this Intel Mac, PyTorch is pinned to its last officially supported x86 macOS release. Model files need about 1.5 GB; allow several more GB for the environment and checkpoints.

```bash
bash scripts/run_local.sh
```

The script installs dependencies into `.venv`, fetches recipes if absent, prepares training examples, trains one epoch, and compares the base model with the adapter. It resumes an interrupted run from the latest checkpoint. It does not re-fetch or rebuild data during resume. Keep the laptop awake and plugged in. On macOS:

```bash
caffeinate -i bash scripts/run_local.sh > outputs-run.log 2>&1
```

Use another terminal to watch progress:

```bash
tail -f outputs-run.log
```

## Chat with the one-recipe model

From this repository, using the existing local environment and weights:

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m beerllm.chat
```

Ask for the complete Helicious Helles recipe, then ask follow-up questions. Type `exit` to stop. A full guide takes several minutes on this CPU. The default selects `outputs/helles-demo-v2/adapter`, its pinned Qwen model and saved teaching/decoding profile. Recipe quantities and source notes are not injected into the prompt.

Add `--prompt "Explain the complete Helicious Helles recipe."` to ask once, or `--base` for the matched untuned model. Oldest complete turns are removed when the context window fills; history is not saved after exit. Earlier adapters remain available through `--adapter PATH`; the broader experiments below are historical.

The source lacks exact water quantities, separate pitching conditions, gelatin dose and carbonation instructions. The model should acknowledge those gaps. The demo needs brewer review before practical use; other recipes and substitutions are unvalidated. A fresh clone contains code and written evidence, while generated weights, datasets and caches remain local ignored artifacts.

## Individual commands for the first API experiment

```bash
# Download the whole PunkAPI catalog, or add --limit 10 for a sample.
python3 scripts/fetch_recipes.py

# Build source-grounded procedure examples and grouped splits.
python3 -m beerllm.dataset

# Train on CPU with LoRA. Configuration: configs/laptop.json.
.venv/bin/python -m beerllm.train

# Resume with the same config, input files and command arguments.
.venv/bin/python -m beerllm.train --resume

# Compare both models on three held-out recipe prompts.
.venv/bin/python -m beerllm.evaluate

# Run tests (the dataset/masking tests do not download models).
.venv/bin/python -m unittest discover -s tests
```

For a short training check before a longer run:

```bash
.venv/bin/python -m beerllm.train --output outputs/my-smoke --max-steps 2 --limit 8 --eval-limit 2
```

Use a new output directory for a new experiment. The trainer refuses to overwrite a previous run; `--resume` checks input hashes and configuration. Checkpoints are saved every 10 optimizer steps, so interruptions may lose work since the last checkpoint. The two most recent checkpoints are retained, only inside that run's output folder.

## What the project does

1. Fetches [PunkAPI v3](https://github.com/alxiw/punkapi) recipes, preserving original fields and units.
2. Rejects missing mash/fermentation fields, invalid ingredient quantities, and selected advanced processes needing specialist review.
3. Builds full ingredient-and-procedure guides. Recipe facts and general brewing guidance are identified separately. Missing exact boil, dry-hop, conditioning and packaging details remain explicit.
4. Groups matching names/ingredient sets before splitting. Training includes both named-recipe requests and recipe-in-context requests. Held-out evaluation supplies the recipe context; it does not pretend to test recall of unseen recipes.
5. Trains LoRA adapters with configurable pinned models and attention/feed-forward targets. The original runner uses SmolLM2-360M; the refined config selects Qwen2.5-1.5B. Only assistant tokens contribute to loss. Overlength examples are counted and skipped, never silently truncated.
6. Saves an adapter, resumable optimizer checkpoints, logs, and base-versus-adapter answers with small automated evaluation proxies.

## Outputs

| Location | Contents |
|---|---|
| `outputs/helles-demo-v2/adapter/` | Default one-recipe LoRA package, tokenizer and fixed chat profile |
| `outputs/helles-demo-eval-v2/` | Frozen selection, matching raw before/after answers and CLI checks |
| `reports/helles-evaluation.json` | Unedited fresh answers, signatures, metrics and case verdicts |
| `data/helles_focus_v18/` | Current 28-conversation supervised rehearsal and frozen hashes |
| `data/instruction_v3/` | Refined conversations, frozen development/final suites and provenance |
| `outputs/qwen-instruction-v3/best-adapter/` | Selected epoch-two Qwen adapter, used explicitly in chat |
| `outputs/qwen-instruction-v3/adapter/` | Preserved final epoch-three adapter |
| `outputs/instruction-v3/` | Matching baseline/adapter answers, metrics and factual inspection |
| `outputs/qwen-instruction-v3/probes/` | Actual factual answers at each saved training checkpoint |
| `data/instruction_v1/` | Eight-recipe teaching experiment, frozen benchmark and dataset hashes |
| `outputs/instruction-lora-v1/adapter/` | Preserved earlier adapter; failed quality checks |
| `outputs/instruction-v1/` | Actual before/after answers, comparison and assistant review |
| `data/punkapi/` | Original API records and field-coverage audit |
| `data/training/` | Train/validation/test JSONL, readable guides, rejected records and dataset report |
| `outputs/beer-lora/adapter/` | Trained LoRA adapter and tokenizer |
| `outputs/beer-lora/checkpoint-*/` | Resumable local checkpoints |
| `outputs/beer-lora/train_metrics.json` | Training runtime and loss |
| `outputs/evaluation.json` | Base/adapter answers, loss and coverage metrics |
| `.cache/huggingface/` | Downloaded base model |

Generated data, models, caches and credentials are ignored by Git. The old GitHub/BeerXML fetcher is removed; any old `data/raw` or `data/curated` files are unused and left alone.

## What is and is not ready

The software supports the entire fetch → curate → train → evaluate → chat workflow. The recipe corpus is still an experimental dataset, not a collection of expert-validated complete brew sheets. API records do not establish every brewing detail. The guides explicitly identify those gaps, rather than inventing timings to make the dataset appear complete. See [source notes](docs/SOURCES.md) and the [evaluation rubric](docs/EVALUATION.md).

This small model may hallucinate ingredients, quantities or process details even after training. An improvement in loss or keyword coverage is not proof that a recipe is accurate or safe to brew. Human review of full procedures is still necessary. Source reuse permissions need checking before redistributing the corpus or derived model; this repository's MIT license covers our code.

## Git workflow

Feature branch → pull request → squash merge into `main`. Keep commit messages concise and one line, authored as Naisarg.
