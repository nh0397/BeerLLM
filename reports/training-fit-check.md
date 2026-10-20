# Diagnosing the failed instructional run

Completed October 19, 2026 (America/Los_Angeles). The previous eight-recipe model still fails the broader brewing-teacher goal. This diagnostic checks a narrower question: can the existing local LoRA training path learn two short factual answers at all?

## Observations

The supervised labels decode to the intended assistant answer, including the expected `<|im_end|>` token. The earlier adapter contains updated, nonzero LoRA weights. No obvious label-mask or end-token error was found. This does not rule out every implementation or optimization issue.

Two follow-ups were selected exclusively from the existing training partition:

- Düsseldorf Altbier: the source has a 3-gallon (11.4 L) kettle volume and a 5.5-gallon (21 L) finished batch.
- Helicious Helles: primary fermentation is 50°F (10°C) for two to three weeks; later lagering is 40°F (4°C) for three to six weeks.

Validation deliberately repeats these same two examples. Its loss is a **training-fit diagnostic, not a generalization measurement**. Neither the original final test set nor another held-out recipe was added to training.

## What was changed

Fresh adapters were trained from the original pinned SmolLM2-360M-Instruct base. The fitting configuration uses two short answers, accumulation of one, learning rate 0.001 and zero LoRA dropout. The earlier failed run used 34 mixed-length conversations, accumulation of four, learning rate 0.0002 and dropout 0.05. These changes were made together, so this is not a causal isolation of any one setting. The diagnostic settings are not a validated prescription for a larger dataset.

| Run | Optimizer steps | Passes over these two examples | Runtime |
|---|---:|---:|---:|
| Initial fitting check | 40 | 20 | 78.8 seconds |
| Extended fitting check | 100 | 50 | 189.2 seconds |

The explicit `--max-steps` overrides the epoch setting. Both runs, their adapters and the previous eight-recipe model are preserved separately. The default chat adapter was not replaced by this two-answer memorization model.

## Actual generated answers

| Question | Base model | 40-step fitting adapter | 100-step fitting adapter |
|---|---|---|---|
| Exact Düsseldorf question | Incorrectly says it is a three-gallon batch | Learns 5.5 gallons but incorrectly gives 19 L | Gives 3 gallons / 11.4 L in the kettle and 5.5 gallons / 21 L finished |
| Exact Helicious question | Treats 40°F or 50°F as interchangeable | Gets main temperature roles, but changes lagering to 3–5 weeks and contradicts the stage order | Preserves the two stages, temperatures and 2–3 / 3–6 week durations |

The extended adapter passed the two exact-prompt factual checks on inspection. Its token-weighted exact-prompt reference loss is approximately 0.0011. That establishes a very limited memorization result, not a reliable beer assistant.

Two fixed rewordings were defined before inspecting the extended adapter's answers:

- “What are the kettle volume and finished batch size for Düsseldorf Altbier?” The model gives generic definitions but omits the requested quantities.
- “Explain the primary fermentation and later lagering temperatures and durations for Helicious Helles.” The model invents different temperatures and durations and confuses the stages.

Both rewordings fail. These questions concern the same trained facts, so they are seen-fact paraphrase diagnostics, not held-out recipe tests. The inspection is by the assistant, not an independent or expert brewer review.

## How to fix the next full experiment

1. **Prove fitting first.** Before a long run, require correct generated quantities and stage relationships on a small training subset. Check the actual answer, not just its loss.
2. **Broaden the teaching tasks.** For reviewed recipes, include full guides, short factual questions, step explanations, comparisons between stages, mistakes and missing-information responses. Use genuinely different question wording and conversational follow-ups; do not count copies as new recipe knowledge. Preserve source facts and audit every derived parameter.
3. **Benchmark a stronger starting model before adapting it.** A candidate is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). Its official card lists 1.54B parameters and a minimum supported Transformers version below our installed version. It has not been downloaded, tested or trained in this diagnostic. Measure CPU runtime, memory and instruction-following on this laptop before selecting it; larger alone is not a proven fix.
4. **Choose training settings with validation.** Check generated answers periodically and balance short factual tasks with long guides. Do not transfer the aggressive two-example fitting settings directly to the full corpus. Anti-repetition decoding can be investigated for usability, but cannot establish correct recipe facts.
5. **Use a fresh final test set.** Separate seen-recipe recall, new wordings of seen facts and interpretation of supplied unseen recipes. Score ingredient–quantity–stage relationships, unsupported claims and procedure completeness. Keep related recipes together and reserve untouched families for final reporting. The original nine-question benchmark has already informed diagnosis and is no longer an untouched final exam for subsequent choices.

The evidence supports improving training coverage and testing the starting model's capabilities. It does not prove that the laptop cannot fine-tune, that LoRA is ineffective or that this small fitting checkpoint is useful beyond its two answers.

## Reproduce

The existing results are saved locally. Choose unused output paths to repeat training/evaluation; existing artifacts are preserved.

```bash
.venv/bin/python -m beerllm.fit_check prepare

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.fit_check evaluate \
  --output outputs/fit-repeat-before.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m beerllm.train \
  --config configs/fit-check.json --data-dir data/fit_check_v1 \
  --max-steps 100 --eval-limit 2 --output outputs/fit-repeat

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.fit_check evaluate \
  --adapter outputs/fit-repeat/adapter --include-paraphrases \
  --output outputs/fit-repeat-after.json
```

The baseline collected for this diagnostic uses the same pinned loader, greedy generation and assistant-token loss helper as the reusable evaluation command. Source questions/references remain frozen under `data/fit_check_v1/`. Actual outputs are `outputs/fit-check-v1-before.json`, `outputs/fit-check-v1-after.json` and `outputs/fit-check-v1-100-after.json`; training artifacts are under `outputs/fit-check-v1/` and `outputs/fit-check-v1-100/`.

All **46 unit tests pass**, with new checks preventing held-out-example selection, silently changed source data or mutation of training examples when generating reworded questions. Python compilation and diff checks pass. No stronger model or broad retraining has been started.
