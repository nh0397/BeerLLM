# One-recipe brewing tutor: Helicious Helles

The previous candidate passes all fifteen independent fitting checks but omits gelatin in one conversation reply (5/6 turns pass). A joint-answer rehearsal is running, retaining those fifteen answers and practicing complete combined replies. The twelve reserved final questions remain untouched; the chat default has not been promoted.

## Source and scope

The lesson uses [AHA’s Helicious Helles](https://homebrewersassociation.org/homebrew-recipe/helicious-helles/): a seven-gallon / 26.5 L batch, eight ingredient entries, grain rests, mash-out, sparging, timed kettle additions, primary fermentation, kegging/gelatin and later lagering. The saved source extraction was checked against its HTML snapshot, SHA-256 `de8d17364a4aeb1f88fa7d504e024ce05ac3a9774958ef5655237c3fe3ebd84e`.

The source omits water volumes and a numeric mineral profile, separate exact pitching conditions, gelatin dose and carbonation/bottling instructions. The teacher should acknowledge these gaps and distinguish them from the supplied temperature and duration ranges. A complete explanation of this source is not a fully specified, expert-approved brewing plan. References and inspections are assistant-authored; no expert brewer has reviewed them.

## Actual checks

| Check | Result | Interpretation |
|---|---:|---|
| Fifteen consumed independent questions | 15/15 | Supervised fitting; every full answer inspected |
| Six consumed actual-history turns | 5/6 | Combined water/gelatin request omits gelatin; rejected |
| Six untouched independent questions | Pending | New wording about the same trained recipe |
| Six untouched actual-history turns | Pending | New wording and each model’s own history |

The consumed questions were explicitly promoted into training after earlier errors. They cannot support held-out accuracy claims. The twelve fresh questions were frozen before that rehearsal, with exact training-question overlap checks. All factual acceptance rules remain in [the rubric](../configs/helles-rubric.json). Correct keywords or low loss alone cannot pass an answer.

## Fixed model and training

- Base: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Ordinary LoRA: rank 16, alpha 32, zero dropout, all seven attention/feed-forward projection types; 18,464,768 trainable parameters.
- Final rehearsal: 15 source-reviewed conversations, four epochs / 60 updates, learning rate 0.0001. It starts from the unchanged saved checkpoint 232 of the preceding 232-example rehearsal.
- Standard assistant-only cross-entropy, full answers, no context truncation during training. All examples fit 2,048 tokens. Frozen validation repeats the trained recipe under another wording.
- Final rehearsal: 1,635.4 seconds wall time, including 1,123.9 seconds of generation checks; peak RSS 15.17 GiB; mean training loss 0.011911.
- The final labels retain fourteen correct replies from the parent and the source-correct completion reference. This is supervised rehearsal, not an independent evaluation.
- Runtime: original teaching system prompt, greedy decoding, recipe facts supplied by trained weights. No recipe lookup, recipe notes or quantities are inserted into the runtime prompt.

The local package is `outputs/helles-demo-v1/adapter`. Its weights are unchanged from `outputs/helles-terminal-v1/adapter`, SHA-256 `15aba46f03d5b143fb115a2cd56dca69e00c3425197b848efe4ba974dd795216`. Rejected checkpoint averages and module subsets are not used. The source adapter, earlier checkpoints, optimizer states, failed answers and datasets remain preserved locally.

## What the metrics mean

**Full-answer pass count** checks correctness, completeness, instruction-following and source gaps together. Read every answer, including its ending. A response that invents a setting, omits a requested stage, answers another question, loops or truncates fails.

**Answer-token cross-entropy** measures how surprising the reference answer is to the model when the correct preceding reference words are supplied. Lower is better for predicting that reference. **Conditional perplexity** is its exponential; a value nearer one means stronger reference prediction. These diagnostics can improve while a generated answer remains wrong, especially when recipe text is memorized. They do not establish brewing accuracy.

The fresh base/adapter pair must use the same pinned base, original system prompt, decoding, suite, loader and evaluator code. Conversation histories differ naturally because each model receives its own previous replies; those histories are not a paired perplexity test.

## Practical limits and evidence

This is a one-recipe learning experiment. New wording on a trained recipe does not establish competence on other recipes, unusual substitutions, scaling or equipment choices. The instruction data and generated guides need brewer review before practical use. CPU generation can take several minutes for a full guide.

[Commands](../docs/SINGLE-RECIPE.md) · [Earlier development conversation](helles-development-conversation.md) · [Preserved experiments and failures](helles-experiments.md). Weights, raw evaluation outputs, data and model caches are local ignored artifacts; a fresh Git clone does not contain them.
