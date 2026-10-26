# One complete recipe before scaling

This experiment teaches only **Helicious Helles**, using the saved AHA source. The target is a complete, understandable recipe plus accurate follow-up answers. It starts from the pinned Qwen2.5-1.5B base, not the previous multi-recipe adapter. Recipe facts are not embedded in the system prompt or silently fetched during evaluation.

The lesson preserves the source's seven-gallon batch, every ingredient, grain rests, hop countdown, primary fermentation, kegging/gelatin and later lagering. Water quantities/profile, a separate pitching temperature, gelatin dose and carbonation instructions are genuinely absent; the model must identify those gaps while retaining the temperature settings that are present. This is a complete explanation of the available source, not an expert-approved brew sheet with every equipment-dependent parameter settled.

The dataset contains 24 conversations: six full-guide requests, twelve direct follow-ups and six conversations. Six epochs give 144 optimizer steps and 36 supervised full-guide presentations. The validation guide repeats the same recipe facts under different wording. None of these counts represent independent recipes or general brewing accuracy.

## Prepare and record the baseline

```bash
.venv/bin/python -m beerllm.recipe_focus prepare

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-focus.json --data-dir data/helles_focus_v1 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v1/development.jsonl \
  --mode base --output outputs/helles-focus-eval-v1/development-before.json
```

## Train and inspect the whole guide

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-focus.json --data-dir data/helles_focus_v1 \
  --eval-limit 1 --probe-suite data/helles_focus_v1/probes.jsonl \
  > outputs-helles-train.log 2>&1
```

The two checkpoints generate an entire training guide and a hop-timer answer. Review the actual answers against [the frozen rubric](../configs/helles-rubric.json). Passing token-pattern checks or low reference loss cannot establish success. Resume an interrupted run using the identical command plus `--resume`, appending to the log. Preserve the existing output; use a new directory for a different experiment. Run only one model job at a time.

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v1/development.jsonl \
  --mode adapter --adapter outputs/helles-focus-v1/adapter \
  --output outputs/helles-focus-eval-v1/development-after.json

.venv/bin/python -m beerllm.benchmark compare --suite data/helles_focus_v1/development.jsonl \
  --before outputs/helles-focus-eval-v1/development-before.json \
  --after outputs/helles-focus-eval-v1/development-after.json \
  --output outputs/helles-focus-eval-v1/development-comparison.json
```

## Repair incomplete follow-ups without restarting the recipe

The first adapter learns the complete guide but scores 3/5 on development because two multi-part questions receive partial answers. The continuation adds varied questions and complete stage lists while preserving the guide and all evaluation bytes. It starts from the original adapter with a fresh optimizer at a lower learning rate. Parent adapter hashes and compatible LoRA settings are checked before training/resume.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-followups.json --data-dir data/helles_focus_v2 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-followups.json --data-dir data/helles_focus_v2 \
  --eval-limit 1 --probe-suite data/helles_focus_v2/probes.jsonl \
  > outputs-helles-followups-train.log 2>&1

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v2/development.jsonl \
  --mode adapter --adapter outputs/helles-followups-v1/adapter \
  --output outputs/helles-followup-eval-v1/development-after.json

.venv/bin/python -m beerllm.benchmark compare --suite data/helles_focus_v2/development.jsonl \
  --before outputs/helles-focus-eval-v1/development-before.json \
  --after outputs/helles-followup-eval-v1/development-after.json \
  --output outputs/helles-followup-eval-v1/development-comparison.json
```

The continuation contains 46 conversations and runs 138 steps. Generated probes check the full guide and both multi-stage follow-ups. The first run, source snapshots and evaluation files are preserved.

## Distinguish known temperatures from missing quantities

The stage-list continuation scores 4/5: it fixes both multi-stage answers, but its two checkpoints falsely deny a supplied temperature when asked about mixed known/unknown settings. The next refinement adds fourteen contrast examples while retaining the working guide and stage questions. Evaluation files remain byte-identical to the original suite.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --config configs/helles-status-refinements.json --output data/helles_focus_v3

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-status.json --data-dir data/helles_focus_v3 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-status.json --data-dir data/helles_focus_v3 --eval-limit 1 \
  > outputs-helles-status-train.log 2>&1

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v3/development.jsonl \
  --mode adapter --adapter outputs/helles-status-v1/adapter \
  --output outputs/helles-status-eval-v1/development-after.json

.venv/bin/python -m beerllm.benchmark compare --suite data/helles_focus_v3/development.jsonl \
  --before outputs/helles-focus-eval-v1/development-before.json \
  --after outputs/helles-status-eval-v1/development-after.json \
  --output outputs/helles-status-eval-v1/development-comparison.json
```

This continuation initializes from `outputs/helles-followups-v1/adapter`, saves in a new directory and trains for two passes over 60 conversations. Full-guide and stage-list rehearsal remains included. The source recipe and old models are preserved.

## Fresh wording and actual conversation

Only after development choices finish, repeat the benchmark pair and comparison with `data/helles_focus_v3/final.jsonl`, the selected continuation adapter and `outputs/helles-status-eval-v1/final-{before,after,comparison}.json`. These are different questions about the same trained recipe, not new recipe families.

The dialogue test feeds each model its own generated answer before asking the next question. Reference answers are saved for review, never inserted into the live history. It can resume an interrupted evaluation with the same output signature.

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.recipe_focus dialogue \
  --protocol data/helles_focus_v3/dialogue.json \
  --output outputs/helles-status-eval-v1/dialogue-before.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.recipe_focus dialogue \
  --protocol data/helles_focus_v3/dialogue.json \
  --adapter outputs/helles-status-v1/adapter \
  --output outputs/helles-status-eval-v1/dialogue-after.json
```

Success requires all five development cases, four final wording cases and four dialogue turns to pass source-based full-answer inspection. It establishes a bounded one-recipe demonstration only. References and inspection remain assistant-authored, not independent expert validation. Results belong in [the run report](../reports/single-recipe-run.md).

After an adapter exists, test normally without supplying recipe notes:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.chat --adapter outputs/helles-status-v1/adapter
```

Ask for the complete Helicious Helles recipe, then ask follow-ups. Use `exit` to stop. The adapter saves its focused system prompt for chat. Old adapters and all collected recipes remain preserved.


## Balanced coverage after a failed reserved test

The status continuation was rejected: development 5/5, reserved wording 1/4. The new run broadens all topics and conversation contexts rather than treating that score as success. It initializes from the first complete-guide adapter and preserves every earlier experiment.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --config configs/helles-balanced-refinements.json --output data/helles_focus_v5
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-balanced.json --data-dir data/helles_focus_v5 --eval-limit 1 \
  > outputs-helles-balanced-train.log 2>&1
```

The original `final.jsonl` is now a regression suite used during selection. `confirmation.jsonl` contains four newly reserved phrasings, frozen before this training run. The original actual-history `dialogue.json` remains reserved. Do not use either reserved set to select a checkpoint. Existing local datasets are frozen; skip preparation when the recorded dataset is already present, and use a new output directory for changed builder/config inputs.
