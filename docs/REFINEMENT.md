# Refined teaching curriculum

The first Qwen attention-adapter trial failed all eight strict development cases. The wider two-answer check can memorize exact prompts but fails rewordings. This refinement addresses response scope and factual-answer coverage before another full run. The [trial report](../reports/instruction-v2-run.md) preserves those negative results.

`data/instruction_v3/` contains 197 conversations from the same eleven source-checked recipes. Added examples pair specific questions with actual source notes, vary direct factual answers, reinforce unknown-private-recipe handling and completion checks, and vary ingredient requests. The system prompt asks for the requested scope rather than ingredients and a full guide on every turn. These are additional teaching contexts, not independent recipes or expert-reviewed brew sheets.

The four validation families and two final families remain excluded from training. Development/final questions and references are retained, with the new system prompt applied consistently. A fresh base evaluation is necessary because both this prompt and the explicit neutral decoder differ from the original trial. Future adapters save their system prompt in `beerllm.json`, which local chat reads automatically. Older adapters retain their existing fallback prompt.

## First check the two-fact curriculum

```bash
.venv/bin/python -m beerllm.refine_instructions
.venv/bin/python -m beerllm.fit_check prepare-varied

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-varied-fit.json --data-dir data/qwen_varied_fit \
  --max-steps 60 --eval-limit 2 > outputs-varied-fit-train.log 2>&1

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.fit_check evaluate \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --data-dir data/qwen_varied_fit \
  --adapter outputs/qwen-varied-fit-v1/adapter --include-paraphrases \
  --output outputs/instruction-v3/varied-fit-after.json
```

That dataset has twelve examples around two existing training facts. Its validation repeats those facts; `evaluation.jsonl` selects the two canonical prompts for the evaluator, which also asks two fixed rewordings. This is a fitting/wording diagnostic, not a test on unfamiliar recipes. The tiny adapter is not promoted as the brewing assistant.

## Candidate full run

Run the larger experiment only after inspecting the small check's generated quantities and stage relationships. The twelve-example check passed both exact prompts and both fixed rewordings on assistant inspection. This covers two trained facts only. The full config adapts attention and feed-forward projections, uses learning rate 0.001 with zero LoRA dropout, and makes three passes over the refined examples. These are empirical choices under evaluation, not a general prescription.

The trainer replays the 88 short factual rows once more per epoch (`short_fact_repeat: 2`). The dataset remains 197 distinct conversations and eleven recipes; training samples 285 rows per epoch, for 855 optimizer steps across three epochs. This matches the approximate per-fact exposure used in the passing fitting check while retaining full guides and source-context tasks. Replays are explicit sampling weights, not additional independent examples.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-instruction-v3.json --data-dir data/instruction_v3 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/instruction_v3/development.jsonl \
  --mode base --output outputs/instruction-v3/development-before.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-instruction-v3.json --data-dir data/instruction_v3 \
  --eval-limit 4 --probe-suite data/instruction_v3/probes.jsonl > outputs-v3-train.log 2>&1

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/instruction_v3/development.jsonl \
  --mode adapter --adapter outputs/qwen-instruction-v3/adapter \
  --output outputs/instruction-v3/development-after.json

.venv/bin/python -m beerllm.benchmark compare --suite data/instruction_v3/development.jsonl \
  --before outputs/instruction-v3/development-before.json \
  --after outputs/instruction-v3/development-after.json \
  --output outputs/instruction-v3/development-comparison.json
```

Use unused output locations for a new experiment. Resume an interrupted training process with identical arguments plus `--resume`; append logs with `>>`. Do not launch simultaneous training jobs. Nine checkpoints and generated probes are retained. Inspect the full answers against [the factual rubric](../configs/instruction-v2-rubric.json); lower perplexity, matched numbers or shorter outputs do not pass the quality gate by themselves.

Once development choices are finished, evaluate the same selected model/base pair with `data/instruction_v3/final.jsonl` and separate `final-before.json` / `final-after.json` files. Do not use those answers for more tuning while continuing to call them an untouched final test.

After a full adapter exists, test it explicitly:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.chat \
  --adapter outputs/qwen-instruction-v3/adapter \
  --recipe-title 'Helicious Helles' \
  --prompt 'Explain only the fermentation and later lagering temperatures and durations.'
```

Omit the recipe-title option to test recall from weights. Supplying the exact saved title tests interpretation with source facts available. Neither mode establishes expert brewing accuracy, and source procedure gaps remain explicit.
