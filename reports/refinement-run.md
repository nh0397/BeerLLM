# Refined Qwen teaching experiment

The [first Qwen trial](instruction-v2-run.md) failed all eight strict development cases despite much lower reference perplexity. Its exact factual prompts also failed. A wider two-answer adapter then memorized two exact answers but failed their rewordings. This experiment addresses scope, response diversity and factual exposure, while preserving those earlier negative results.

## Passed small fitting gate

Twelve examples teach two existing training facts through different questions, direct-answer variants, prior-conversation context and supplied source notes. A scope-aware system prompt requests only the answer asked for. Qwen2.5-1.5B-Instruct uses LoRA on attention and feed-forward projections, 18,464,768 trainable parameters, learning rate 0.001, no dropout and neutral greedy decoding.

The 60-step / five-epoch fitting run took 647.3 seconds (10.8 minutes), with 15.46 GiB peak process RSS. Both exact questions and both fixed rewordings preserve the requested factual relationships on assistant inspection:

- Düsseldorf: 3 gallons / 11.4 L concentrated kettle volume, then 5.5 gallons / 21 L after topping up in the fermenter.
- Helicious: primary fermentation at 50°F / 10°C for two to three weeks, then later lagering at 40°F / 4°C for three to six weeks.

These are two trained facts, not unfamiliar recipes or a population accuracy estimate. The tiny adapter is not promoted. Multiple data/prompt/training changes occurred together; the result does not isolate one cause. Full outputs and the inspection are saved in `outputs/instruction-v3/varied-fit-after.json` and `varied-fit-review.json`.

## Full experiment plan

The frozen `data/instruction_v3/` release has 197 distinct conversations across eleven source-checked training recipes, with four validation families and two reserved final families. References are assistant-authored/source-checked, not expert brewer-reviewed. Source procedure gaps and typos remain explicit.

Short factual examples get explicit replay weighting: the 88 short rows appear twice per epoch. The trainer therefore samples 285 rows per epoch, while the dataset still contains 197 distinct conversations and eleven recipes. Three epochs give 855 optimizer steps and roughly match the passing small check's per-fact exposure. Full guides, ingredients, source-specific Q&A, conversational follow-ups and general completion/uncertainty checks remain included. No examples exceed the 2,048-token training budget.

The full model starts from pinned Qwen revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; the fitting adapter is not reused. The config retains the small check's rank 16, alpha 32, learning rate 0.001, zero dropout and attention/feed-forward targets. It saves nine checkpoints with generated development probes, plus loss checks on all four validation examples. Optimizer state and all previous experiments are preserved.

A fresh base evaluation uses the exact revised system prompt and neutral decoder that will be used after training. Existing older-prompt/penalized results are not relabeled as a matched control. After development choices finish, the same selected model/base pair will be evaluated on the two reserved final families with source notes supplied. They have not been evaluated yet.

## Fresh baseline

One of eight development cases passes the frozen factual rubric on assistant inspection: the unavailable private-recipe request correctly asks for notes. Named recipe recall fails, the supplied Ardley procedure corrupts hop timing/yeast identification, and the completion check invents a universal final-gravity threshold. This is a small convenience sample, not a population accuracy estimate.

| Task | Before reference perplexity |
|---|---:|
| Seen facts under new wording (four cases) | 9.742 |
| Complete seen-recipe guide | 7.816 |
| Supplied validation recipe | 2.128 |
| Unavailable private recipe | 14.957 |
| Completion check | 66.175 |

Two of eight answers hit their output-token limits. Peak base process RSS is 8.86 GiB. Raw answers and the inspection are preserved under `outputs/instruction-v3/`.

## Epoch-one observation

The first 285 optimizer steps are complete. Validation loss is 0.188804 across the four supplied-recipe references. All three periodic probes still fail the strict factual check. The Helles answer now retrieves both correct temperature/duration pairs, but incorrectly places primary fermentation after kegging and invents a clarification temperature. Volume questions still mix recipe quantities with unrelated CO2/priming statements. This is partial value learning, not a passing brewing instruction.

Full checkpoint answers and inspection notes are preserved under `outputs/qwen-instruction-v3/probes/` and `outputs/instruction-v3/probe-review.json`. Training continues with the fixed data, replay weights and three-epoch plan.

At checkpoint 475, both fixed reworded probes preserve the correct quantities, units and stage relationships for the first time. The exact volume prompt recovers the correct numbers but appends the confusing unsupported sentence “It does not boil the kettle.” It therefore still fails full-answer review. These probes cover only two trained facts; broader development and final answers remain untested for this adapter.

## Epoch-two observation

At step 570, all three factual probes pass on full-answer inspection. The exact and reworded volume questions preserve 3 gallons / 11.4 L in the kettle and 5.5 gallons / 21 L after dilution. The Helles rewording preserves primary fermentation at 50°F / 10°C for two to three weeks and later lagering at 40°F / 4°C for three to six weeks after kegging/clarification. The exact answer repeats “in the fermenter” harmlessly; it does not add a conflicting instruction.

Validation loss is 0.157951 across four recipe references, down from 0.188804 at epoch one. Neither this loss nor the three probes establish complete recipe quality. The final pass continues with the same frozen configuration.

Improvement is not monotonic. At checkpoint 665, the reworded volume answer regresses to unrelated hop quantities and boil timers, while the other two probes pass. At checkpoint 760, all three pass again. The final full development set, rather than a cherry-picked early checkpoint, will determine the broader result.

**Status:** two of three epochs complete; final pass running; after/final evaluation pending. No full adapter has been promoted.

## Reproduction

See [the exact commands](../docs/REFINEMENT.md). Local generated data, weights, complete answers and logs are ignored by Git. The trainer saves the system prompt with the adapter so chat can use the same instruction. Run only one training process at a time; resume an interrupted run with matching inputs/configuration.

All 60 unit tests pass, including masking, source/context integrity, held-out-family guards, neutral decoding, varied-fit selection, replay weights, prompt metadata and chat history checks. Python compilation and diff checks pass. Actual generated-answer inspection remains necessary; loss, perplexity and replay counts do not establish brewing accuracy.
