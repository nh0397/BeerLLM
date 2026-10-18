# BeerLLM

A local LoRA fine-tuning experiment for beer recipes and brewing procedures. Runs on an Intel Mac with 32 GB RAM using a small instruction model. No cloud account or paid API is needed.

**First run:** training and evaluation completed locally. The adapter returned recipe JSON instead of brewing procedures in all three evaluated examples. The workflow is operational, but the model does not yet meet the brewing-instruction goal. See the [first-run report](reports/first-run.md).

## AHA instruction-data pipeline

The next dataset uses actual written recipe directions from selected public AHA pages. See the [roadmap](docs/ROADMAP.md) for milestones and acceptance checks.

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m beerllm.aha --urls-file configs/aha-sample.txt
.venv/bin/python -m beerllm.aha_dataset
.venv/bin/python -m beerllm.train --data-dir data/aha/training_candidates --check-data
```

This extracts source instructions, exports natural-language conversation candidates, and checks tokenization without training. Data is saved separately under `data/aha/`. Review the candidate answers before using them in a long run. The existing one-command runner below still reproduces the older PunkAPI experiment; it does not automatically switch to AHA data.

## Quick start

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

## Chat after training

```bash
.venv/bin/python -m beerllm.chat --prompt "Give me the recipe and brewing procedure for Punk IPA 2007 - 2010."
```

That uses model weights only. Named recipe recall may be inaccurate. To explicitly supply a downloaded recipe as context:

```bash
.venv/bin/python -m beerllm.chat --recipe-id 1 --prompt "Explain this recipe and its brewing procedure."
```

Use `--base` to compare the original model. Omit `--prompt` for an interactive terminal. Each question is independent; chat history is not retained.

## Individual commands

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
5. Trains `HuggingFaceTB/SmolLM2-360M-Instruct` with LoRA on query/value projections. Only assistant tokens contribute to loss. Overlength examples are counted and skipped, never silently truncated.
6. Saves an adapter, resumable optimizer checkpoints, logs, and base-versus-adapter answers with small automated evaluation proxies.

## Outputs

| Location | Contents |
|---|---|
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
