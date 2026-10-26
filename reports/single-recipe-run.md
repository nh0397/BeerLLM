# Helicious Helles: complete-guide fitting experiment

**Status: focused training is running. The halfway checkpoint passes its complete-guide and hop-timer training probes; fresh-wording and actual-history evaluation remain pending.**

The [previous experiment](refinement-run.md) improved short factual answers but failed complete guides. This experiment makes the full guide the central gate and limits scope to one recipe before any further scaling.

The selected source is [AHA's Helicious Helles](https://homebrewersassociation.org/homebrew-recipe/helicious-helles/), from the saved snapshot with HTML SHA-256 `de8d17364a4aeb1f88fa7d504e024ce05ac3a9774958ef5655237c3fe3ebd84e`. Re-extraction matches the saved ingredient list and directions. This verifies provenance, not expert brewing correctness. Actual omissions are preserved rather than filled with invented settings.

The authored lesson has 24 conversations: six complete-guide requests, twelve direct follow-ups and six conversations. Every row uses the same trained recipe. The system prompt names the recipe and sets response expectations, but contains no recipe quantities. Training references supervise 5,410 answer tokens; the longest complete conversation is 880 tokens. No examples exceed the 2,048-token budget or are truncated.

The proposed run starts from pinned Qwen2.5-1.5B-Instruct with rank-16 LoRA on attention/feed-forward projections, learning rate 0.0003, zero dropout and six epochs (144 steps). That provides 36 supervised complete-guide presentations. Validation repeats the trained facts under a different question; it is not an unfamiliar-recipe estimate. Multiple curriculum/training changes occur together, so this experiment cannot isolate one cause of improvement.

Before generation, [the factual rubric](../configs/helles-rubric.json) fixes the required ingredient quantities, all temperature/time relationships, hop countdown, genuine gaps, plain explanations and scope control. Success requires five development cases, four reserved wording cases and four actual-history dialogue turns to pass complete-answer inspection. The dialogue uses the model's own preceding answer, not a reference guide placed into history. All evaluations concern the same trained recipe.

All 63 software tests pass, including new checks for source/family restrictions, frozen outputs, question separation and interrupted dialogue resumption without reference-answer leakage. The [commands](../docs/SINGLE-RECIPE.md) reproduce the setup. Data, weights, full generated answers and logs stay local and ignored by Git. Previous models, recipe data and checkpoints are preserved.

## Before-training result

All five development cases fail full-answer source inspection. The complete-guide answer substitutes a generic extract recipe and wrong fermentation settings; the ingredient answer invents flavorings and a 750 ml batch. Mash settings are wrong and truncated. The cold-stage answer misses the source's exact primary/lagering settings and gelatin order. The final question falsely denies the source's stated fermentation temperatures. Two of five outputs hit their length limits.

This baseline uses the same focused system prompt and questions planned for the after run, with neutral greedy decoding and no supplied recipe notes. Full output and inspection are local in `outputs/helles-focus-eval-v1/development-before.json` and `review.json`. Training has started from the base weights; the previous multi-recipe adapter is not reused.

## Halfway generated-answer check

At step 72 / epoch three, the exact training guide passes all full-guide rubric criteria on assistant source inspection. It preserves every source ingredient entry, the seven-gallon batch, successive grain rests, the 90-minute boil and 30/80-minute elapsed additions, cooling/pitching distinction, primary fermentation, kegging/gelatin and later lagering. It acknowledges actual missing settings without denying the provided temperatures, and explains the main brewing terms plainly. The hop-timer follow-up also passes. Neither answer is truncated or loops.

Both generated texts exactly match their authored references. That establishes deliberate fitting on these training prompts; it does not establish success on the reserved wording or on a real conversation. The fixed six-epoch run continues. Complete answers are preserved in `outputs/helles-focus-v1/probes/step-72.json`.
