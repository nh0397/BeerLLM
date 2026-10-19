# Train and compare the instructional model

This is a small learning experiment with eight source-checked recipes and handwritten teaching plans, not a fine-tune on all 1,498 collected records. The 34 training conversations include eight named-recipe guides, eight supplied-recipe guides, 16 follow-ups and two general brewing checks. Original source references remain attached. No expert brewer has reviewed the targets.

The model is the same pinned SmolLM2-360M-Instruct checkpoint as the first experiment. LoRA trains query/value adapters at rank eight for five epochs on CPU. Keeping the base model fixed makes the before/after comparison easier to interpret. The earlier adapter is preserved.

## Run in this order

From the repository root, use the existing `.venv`:

```bash
# Build teaching conversations and freeze the evaluation questions.
.venv/bin/python -m beerllm.instruction_dataset
.venv/bin/python -m beerllm.benchmark prepare

# Check context lengths and assistant-only supervision without training.
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/instruction-laptop.json --data-dir data/instruction_v1 --check-data

# BEFORE: save the unmodified base model's actual answers.
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --mode base --output outputs/instruction-v1/before.json

# Train a separate adapter. Do not run another training process alongside it.
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/instruction-laptop.json --data-dir data/instruction_v1

# AFTER: exactly the same questions, source facts, references and greedy decoding.
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --mode adapter --output outputs/instruction-v1/after.json

.venv/bin/python -m beerllm.benchmark compare \
  --before outputs/instruction-v1/before.json \
  --after outputs/instruction-v1/after.json \
  --output outputs/instruction-v1/comparison.json
```

`caffeinate` is macOS-specific; omit it on other operating systems. Offline mode uses the already cached model. The first installation/download still follows the main README.

An interrupted training run resumes with the same command plus `--resume`. Checkpoints are saved every five optimizer steps; the two most recent are retained inside this new run's output folder. Resume requires the same data, configuration and limits. A completed evaluation returns its existing report; an interrupted evaluation resumes from completed cases. A changed suite, evaluator or adapter requires a new evaluation output path.

Data or lesson changes require a new dataset output directory; supply that directory consistently to the dataset builder, benchmark preparation and trainer. The benchmark checks recipe-family partitions before writing its suite. Do not tune training choices against the final test answers; any later attempt informed by those answers needs a new untouched final test set.

## Try the resulting adapter

```bash
.venv/bin/python -m beerllm.chat --adapter outputs/instruction-lora-v1/adapter \
  --prompt "Explain the recipe and brewing steps for Alt Ale Altbier."
```

Omit `--prompt` for terminal chat, or add `--base` to compare the unmodified model. Terminal chat retains the conversation in memory until you exit, so you can ask follow-ups. It removes the oldest complete turns if needed to fit the context window and tells you when it does so. It does not save chat history to disk. To test interpretation of another recipe, put its facts into your prompt explicitly. The adapter cannot reliably know a private or unseen recipe from its name alone.

The local benchmark reports contain full answers, references, per-check matches, conditional answer loss/perplexity, response times, repetition and token-limit flags. Read the answers alongside the scores. The [metric guide](METRICS.md) explains the names and limitations.
