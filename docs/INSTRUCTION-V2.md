# Larger-model teaching experiment

This is a local CPU LoRA experiment on the Intel Mac with 32 GB RAM. It uses the pinned Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; the [official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) documents its 1.54B parameters and loading requirements. Our installed Transformers 4.46.3 supports its architecture. No cloud service or paid training is used.

The first capability probe fits in memory but still invents source parameters and private-recipe details. Model size alone has not fixed the assistant.

The current laptop already has the pinned model cached. On a new machine, leave `HF_HUB_OFFLINE` unset for the first model command so the loader can download its weights.

## Dataset

The builder makes a separate frozen `data/instruction_v2/` release. The old data and adapters remain intact.

- 138 training conversations from eleven source-checked AHA recipes: the earlier eight plus Columbus Pale Ale, Fuller's Chiswick Bitter Clone and Your First SMaSH Ale.
- Named and supplied-recipe guides, ingredients, missing settings, varied factual questions, and multi-turn follow-ups. Only the final assistant answer contributes to training loss; prior replies are context.
- Four pre-existing validation families for development loss. Eight generation cases separately examine seen-recipe recall under new wording, a full guide, a supplied validation recipe, private-recipe uncertainty and completion checks.
- Two newly reserved final recipe families, Anchor Steam Clone and Holier than Meow Hefeweizen. Neither enters training or periodic probes. Their source facts are supplied in final prompts, so this measures interpreting unfamiliar recipes, not remembering unseen facts.
- All seventeen source HTML hashes and re-extracted ingredients/directions are verified. Written lessons are checked against those source records by the assistant, not an expert brewer. Source typos and missing instructions stay explicit. Prompt variants reuse facts and do not represent additional independent recipes.

The old nine-question benchmark has informed previous decisions and is not used as a fresh final test.

The recorded original before/after pair used the inherited Qwen repetition penalty of 1.1. Current code explicitly uses 1.0 and records all greedy settings. Preserve the original results and choose unused output paths for a new comparison; [commit 70f3a9a](https://github.com/nh0397/BeerLLM/tree/70f3a9a) preserves the earlier generation code. See [the refinement experiment](REFINEMENT.md) for the new prompt/curriculum.

## Commands

```bash
# Build once; repeated builds verify hashes and refuse changed inputs.
.venv/bin/python -m beerllm.instruction_v2

# Check masking/context lengths without loading model weights.
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-instruction-laptop.json \
  --data-dir data/instruction_v2 --check-data

# Freeze base answers before adapting this model.
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --suite data/instruction_v2/development.jsonl --mode base \
  --output outputs/instruction-v2/development-before.json

# Run one training process only. Six checkpoints and their probe answers are retained.
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-instruction-laptop.json --data-dir data/instruction_v2 \
  --eval-limit 4 --probe-suite data/instruction_v2/probes.jsonl > outputs-v2-train.log 2>&1

# Resume only an interrupted run, with identical arguments plus --resume.
# Append its log with >> instead of replacing it with >.
# Choose an unused output directory for a new experiment instead.

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --suite data/instruction_v2/development.jsonl --mode adapter \
  --adapter outputs/qwen-instruction-v2/adapter \
  --output outputs/instruction-v2/development-after.json

.venv/bin/python -m beerllm.benchmark compare \
  --suite data/instruction_v2/development.jsonl \
  --before outputs/instruction-v2/development-before.json \
  --after outputs/instruction-v2/development-after.json \
  --output outputs/instruction-v2/development-comparison.json
```

Once development decisions are finished, repeat the same base/adapter benchmark commands with `--suite data/instruction_v2/final.jsonl` and separate `final-before.json` / `final-after.json` output paths, then compare that pair with the final suite. Do not tune settings from those final answers and still call them an untouched test.

The builder also freezes three periodic probes: one exact training question and two new-wording development questions. Their generated answers are saved under `outputs/qwen-instruction-v2/probes/`. These are inspected to distinguish fitting from generalization; loss alone does not pass the experiment.

To test the new adapter explicitly after it exists:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.chat \
  --adapter outputs/qwen-instruction-v2/adapter \
  --prompt 'For Helicious Helles, explain fermentation and later lagering as separate stages.'
```

The default adapter is not changed merely because this training finishes. A model must pass generated-answer review before promotion. Evaluate ingredient/quantity/stage relationships, omissions, unsupported claims, understandable explanations and scope. Perplexity is compared within the same pinned model, prompts and reference tokens; raw perplexity across different tokenizers is not a model ranking. Token-limit hits and process memory/runtime are operational measurements, not brewing accuracy.

To test explanation with actual AHA source notes supplied explicitly:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.chat \
  --adapter outputs/qwen-instruction-v2/adapter --recipe-title 'Helicious Helles' \
  --prompt 'Explain only the fermentation and later lagering stages, including their temperatures and durations.'
```

The title selects a saved instruction candidate exactly and prints its source URL. Omit `--recipe-title` to test recall from weights. Chat defaults to a practical 2,048-token context budget on this laptop, removing whole older turns as needed; `--context-limit` can explicitly change that budget within the model's native window.

For a live progress check in another terminal:

```bash
tail -f outputs-v2-train.log
```

A changing optimizer-step counter and finite loss show that optimization is progressing. `outputs/qwen-instruction-v2/probes/step-*.json` contains actual generated answers at each save. After completion, `train_metrics.json` records runtime and peak process memory. Those checks show execution health; answer review determines whether the model is useful.

## Fitting gate after the attention-only pilot

The completed attention-adapter run still fails exact factual recall. Check whether a wider adapter can fit two short training answers before selecting another full run. This tiny checkpoint is a diagnostic and is not promoted as a beer assistant.

```bash
.venv/bin/python -m beerllm.fit_check prepare \
  --source data/instruction_v2/train.jsonl --output data/qwen_fit_check

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-fit-check.json --data-dir data/qwen_fit_check \
  --max-steps 80 --eval-limit 2 > outputs-qwen-fit-train.log 2>&1

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.fit_check evaluate \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --data-dir data/qwen_fit_check \
  --adapter outputs/qwen-fit-check-v1/adapter --include-paraphrases \
  --output outputs/instruction-v2/qwen-fit-after.json
```

This checks two exact training prompts and two new wordings of the same facts, not held-out recipe knowledge. The higher learning rate and repeated exposure are part of the fitting setup, not a validated prescription for the full dataset.
