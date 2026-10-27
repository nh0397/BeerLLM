# One-recipe brewing tutor: Helicious Helles

The frozen model passes all twelve fresh checks on its first reserved evaluation: six independent questions and a six-turn conversation using its own replies. It also retains all seventeen fitting inputs and all six consumed history turns. Matched untuned-base controls and the final terminal integration check are still running.

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

[Commands](../docs/SINGLE-RECIPE.md) · [Actual fresh conversation](helles-conversation.md) · [Earlier development conversation](helles-development-conversation.md) · [Preserved experiments and failures](helles-experiments.md). Weights, raw evaluation outputs, data and model caches are local ignored artifacts; a fresh Git clone does not contain them.
