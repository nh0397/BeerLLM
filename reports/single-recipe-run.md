# One-recipe brewing tutor: Helicious Helles

The one-recipe candidate passes all twelve fresh checks on its first reserved evaluation: six independent questions and a six-turn conversation using its own replies. The matched untuned base passes zero of these twelve full-answer checks. It is the default local chat model. A real two-turn terminal chat reproduces the complete fresh guide and follow-up exactly, using no adapter override or recipe-context flags. All 77 software tests, compilation and shell syntax checks pass.

## Source and scope

The lesson uses [AHA’s Helicious Helles](https://homebrewersassociation.org/homebrew-recipe/helicious-helles/): a seven-gallon / 26.5 L batch, eight ingredient entries, grain rests, mash-out, sparging, timed kettle additions, primary fermentation, kegging/gelatin and later lagering. The saved source extraction was checked against its HTML snapshot, SHA-256 `de8d17364a4aeb1f88fa7d504e024ce05ac3a9774958ef5655237c3fe3ebd84e`.

The source omits water volumes and a numeric mineral profile, separate exact pitching conditions, gelatin dose and carbonation/bottling instructions. The teacher should acknowledge these gaps and distinguish them from the supplied temperature and duration ranges. A complete explanation of this source is not a fully specified, expert-approved brewing plan. References and inspections are assistant-authored; no expert brewer has reviewed them.

## Actual checks

| Check | Result | Interpretation |
|---|---:|---|
| Seventeen consumed fitting inputs | 17/17 | Fifteen independent inputs plus joint quantities with/without history |
| Six consumed actual-history turns | 6/6 | Uses the model’s own previous replies |
| Six reserved independent questions | 6/6 | New wording about the same trained recipe |
| Six reserved actual-history turns | 6/6 | New wording and the model’s own history |

## Matched before/after results

| Measure | Untuned base | Fine-tuned model |
|---|---:|---:|
| Fresh independent full-answer passes | 0/6 | 6/6 |
| Fresh actual-history full-answer passes | 0/6 | 6/6 |
| Responses hitting their token limit | 11/12 | 0/12 |
| Reference answer-token cross-entropy | 2.5530 | 0.04517 |
| Conditional reference perplexity | 12.8461 | 1.0462 |

Cross-entropy/perplexity use the same 1,142 reference answer tokens from the six independent questions. The base controls were generated after training, using the unchanged pinned base weights and the same prompt, greedy policy, budgets, loader and evaluator as the frozen adapter. “Before” names the untuned model state. Conversation turns use each model’s own history and are inspected separately.

The base invents recipe quantities and repeats equipment/ingredient text. Its completion answer suggests measuring pH with a hydrometer; its definition answer incorrectly calls boiling mashing. Its sanitation answer overstates sanitizing as killing all microbes. Some basic definitions are partly correct, but no full response passes the fixed rubric. Truncation also contributes to its failures. These results apply to this recipe, wording set and decoding policy; they are not a general ranking of base-model brewing ability.

Read [all unedited before/after answers and case-level verdicts](helles-evaluation.json) and [the actual fresh conversation](helles-conversation.md).

The consumed questions were explicitly promoted into training after earlier errors. They cannot support held-out accuracy claims. The twelve fresh questions were frozen before that rehearsal, with exact training-question overlap checks. All factual acceptance rules remain in [the rubric](../configs/helles-rubric.json). Correct keywords or low loss alone cannot pass an answer.

## Fixed model and training

- Base: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Ordinary LoRA: rank 16, alpha 32, zero dropout, all seven attention/feed-forward projection types; 18,464,768 trainable parameters.
- Final rehearsal: 28 source-reviewed conversations, four epochs / 112 updates, learning rate 0.00005. It retains fifteen correct independent answers, six consumed history turns and seven joint-quantity examples, starting from the completed terminal adapter.
- Standard assistant-only cross-entropy, full answers, no context truncation during training. All examples fit 2,048 tokens. Frozen validation repeats the trained recipe under another wording.
- Final rehearsal: 2,398.2 seconds wall time, including 1,132.1 seconds of generation checks; peak RSS 18.13 GiB; mean training loss 0.057423.
- The combined-quantity answer explicitly addresses gelatin and water separately. All final training labels were source-checked; this is supervised rehearsal, not independent evaluation.
- Runtime: original teaching system prompt, greedy decoding, recipe facts supplied by trained weights. No recipe lookup, recipe notes or quantities are inserted into the runtime prompt.

The local package is `outputs/helles-demo-v2/adapter`. Its weights are unchanged from `outputs/helles-joint-v1/adapter`, SHA-256 `3909cf83dbd5182dd5d6a686b6c86af1b66ca6f4ec06d8d41fe2e753b55d6860`. Rejected checkpoint averages and module subsets are not used. The source adapter, earlier checkpoints, optimizer states, failed answers and datasets remain preserved locally.

## What the metrics mean

**Full-answer pass count** checks correctness, completeness, instruction-following and source gaps together. Read every answer, including its ending. A response that invents a setting, omits a requested stage, answers another question, loops or truncates fails.

**Answer-token cross-entropy** measures how surprising the reference answer is to the model when the correct preceding reference words are supplied. Lower is better for predicting that reference. **Conditional perplexity** is its exponential; a value nearer one means stronger reference prediction. These diagnostics can improve while a generated answer remains wrong, especially when recipe text is memorized. They do not establish brewing accuracy.

The fresh base/adapter pair must use the same pinned base, original system prompt, decoding, suite, loader and evaluator code. Conversation histories differ naturally because each model receives its own previous replies; those histories are not a paired perplexity test.

## Practical limits and evidence

This is a one-recipe learning experiment. New wording on a trained recipe does not establish competence on other recipes, unusual substitutions, scaling or equipment choices. The instruction data and generated guides need brewer review before practical use. CPU generation can take several minutes for a full guide.

[Commands](../docs/SINGLE-RECIPE.md) · [Actual fresh conversation](helles-conversation.md) · [Earlier development conversation](helles-development-conversation.md) · [Preserved experiments and failures](helles-experiments.md). Weights, raw evaluation outputs, data and model caches are local ignored artifacts; a fresh Git clone does not contain the model weights or full generated corpora. The final source-reviewed rehearsal labels and selected unedited evaluation answers are tracked as text evidence.

## Use the saved model

From the repository root:

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -m beerllm.chat
```

Type `exit` to finish. The measured fresh full guide takes 217.7 seconds (3.6 minutes); the median independent answer takes 28.2 seconds on this Intel CPU.
