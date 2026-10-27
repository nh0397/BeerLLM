# Helicious Helles: run and evaluate

The trained one-recipe candidate is saved locally at `outputs/helles-demo-v2/adapter`. It passes the twelve fresh wording/conversation checks; matched base controls and the final integration check are still running. See [the current report](../reports/single-recipe-run.md) for measured status and source gaps.

## Try the candidate after the active model job finishes

Run from this repository’s root:

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m beerllm.chat \
  --adapter outputs/helles-demo-v2/adapter
```

Ask for the whole Helicious Helles recipe, then ask follow-ups. Type `exit` to finish. Recipe quantities and source notes are not added to the prompt. The saved profile supplies the pinned model and original teaching instruction. A full guide can take several minutes on the Intel CPU. Weights and caches are already local; a fresh clone does not include them.

## Reproduce the current terminal training

The existing frozen `data/helles_focus_v18` contains 28 **consumed** teaching conversations with source-reviewed practice labels. The parent checkpoint and all preceding curricula remain local. Preserve the existing outputs. Resume an interrupted original run with `--resume` and the same arguments; use `--output` with a new directory for a separate run.

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-joint.json --data-dir data/helles_focus_v18 \
  --eval-limit 1 --probe-suite data/helles_focus_v18/probes.jsonl \
  --output outputs/helles-joint-reproduction
```

The original final run uses `outputs/helles-joint-v1`. Its source-reviewed labels are tracked at `configs/helles-joint-rehearsal.jsonl`; model facts are learned from assistant answers with user/system tokens masked from the loss. Earlier curriculum preparation, exact commands and failures are preserved in [the experiment command history](HELLES-EXPERIMENTS.md) and [experiment report](../reports/helles-experiments.md).

## Fresh matched evaluation

Run one model job at a time. The final reservation consists of six independent questions and six actual-history turns, frozen before the terminal rehearsal. Use a new output name for different weights, code or policy; the evaluator refuses incompatible resumes.

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --suite configs/helles-fresh-confirmation.jsonl --mode base \
  --decoding-config configs/helles-greedy-decoding.json \
  --output outputs/helles-demo-eval-v2/fresh-before.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --suite configs/helles-fresh-confirmation.jsonl --mode adapter \
  --adapter outputs/helles-demo-v2/adapter \
  --decoding-config configs/helles-greedy-decoding.json \
  --output outputs/helles-demo-eval-v2/fresh-after.json

.venv/bin/python -m beerllm.benchmark compare \
  --suite configs/helles-fresh-confirmation.jsonl \
  --before outputs/helles-demo-eval-v2/fresh-before.json \
  --after outputs/helles-demo-eval-v2/fresh-after.json \
  --output outputs/helles-demo-eval-v2/fresh-comparison.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m beerllm.recipe_focus dialogue \
  --protocol configs/helles-fresh-dialogue.json \
  --decoding-config configs/helles-greedy-decoding.json \
  --output outputs/helles-demo-eval-v2/fresh-dialogue-before.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m beerllm.recipe_focus dialogue \
  --protocol configs/helles-fresh-dialogue.json \
  --adapter outputs/helles-demo-v2/adapter \
  --decoding-config configs/helles-greedy-decoding.json \
  --output outputs/helles-demo-eval-v2/fresh-dialogue-after.json
```

Full-answer inspection against [the source rubric](../configs/helles-rubric.json) determines passing. Loss, perplexity and keyword matches do not establish brewing accuracy. Reference answers are saved for review and never inserted into conversational history.

## Software checks

```bash
TMPDIR="$PWD/outputs/test-tmp" .venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall -q beerllm scripts tests
bash -n scripts/run_local.sh
```
