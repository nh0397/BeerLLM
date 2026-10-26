# Helicious Helles: complete-guide fitting experiment

**Status: setup and source checks complete; before-training evaluation running. No new model success is claimed yet.**

The [previous experiment](refinement-run.md) improved short factual answers but failed complete guides. This experiment makes the full guide the central gate and limits scope to one recipe before any further scaling.

The selected source is [AHA's Helicious Helles](https://homebrewersassociation.org/homebrew-recipe/helicious-helles/), from the saved snapshot with HTML SHA-256 `de8d17364a4aeb1f88fa7d504e024ce05ac3a9774958ef5655237c3fe3ebd84e`. Re-extraction matches the saved ingredient list and directions. This verifies provenance, not expert brewing correctness. Actual omissions are preserved rather than filled with invented settings.

The authored lesson has 24 conversations: six complete-guide requests, twelve direct follow-ups and six conversations. Every row uses the same trained recipe. The system prompt names the recipe and sets response expectations, but contains no recipe quantities. Training references supervise 5,410 answer tokens; the longest complete conversation is 880 tokens. No examples exceed the 2,048-token budget or are truncated.

The proposed run starts from pinned Qwen2.5-1.5B-Instruct with rank-16 LoRA on attention/feed-forward projections, learning rate 0.0003, zero dropout and six epochs (144 steps). That provides 36 supervised complete-guide presentations. Validation repeats the trained facts under a different question; it is not an unfamiliar-recipe estimate. Multiple curriculum/training changes occur together, so this experiment cannot isolate one cause of improvement.

Before generation, [the factual rubric](../configs/helles-rubric.json) fixes the required ingredient quantities, all temperature/time relationships, hop countdown, genuine gaps, plain explanations and scope control. Success requires five development cases, four reserved wording cases and four actual-history dialogue turns to pass complete-answer inspection. The dialogue uses the model's own preceding answer, not a reference guide placed into history. All evaluations concern the same trained recipe.

All 63 software tests pass, including new checks for source/family restrictions, frozen outputs, question separation and interrupted dialogue resumption without reference-answer leakage. The [commands](../docs/SINGLE-RECIPE.md) reproduce the setup. Data, weights, full generated answers and logs stay local and ignored by Git. Previous models, recipe data and checkpoints are preserved.
