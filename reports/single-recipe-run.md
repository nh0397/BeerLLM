# Helicious Helles: complete-guide fitting experiment

**Status: first training run and development review complete (3/5 after versus 0/5 before). The complete guide works under fresh wording; two follow-ups omit requested stages. The continuation has finished; its final checkpoint scores 4/5 on development. An earlier checkpoint is being evaluated before selection. Final wording and actual-history tests remain unopened.**

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

## Completed training

All 144 optimizer steps / six epochs finished in 2,481.4 seconds (41.4 minutes). Mean training loss is 0.398605; peak process RSS is 13.80 GiB. All 24 examples fit without truncation or overlength skips. The final guide and timer probes again exactly match the source-checked references. Both checkpoints (72 and 144), the final adapter and previous experiments are preserved. The final step-144 adapter is the candidate for development evaluation.

| Epoch | Full-guide reference loss, different wording |
|---|---:|
| 1 | 0.667612 |
| 2 | 0.024986 |
| 3 | 0.012722 |
| 4 | 0.002817 |
| 5 | 0.003171 |
| 6 | 0.002074 |

The adapter's saved system prompt and training-data hash match the frozen lesson. Validation still repeats this one recipe's facts; low loss and exact training-prompt output do not replace the upcoming fresh-question and conversation reviews.

## Fresh development wording: correct guide, incomplete follow-ups

The first adapter passes three of five development cases, versus zero of five for the matched base. Its complete guide and ingredient list exactly match the source-checked references under new questions. The known-versus-missing answer correctly preserves both temperature stages and acknowledges absent quantities.

Two cases fail completeness: the mash question returns only the two rests, omitting the explicitly requested mash-out and sparging settings; the after-boil question returns only kegging/lagering, omitting cooling, yeast and primary fermentation. A development-only generic instruction to enumerate all requested stages does not repair either answer. Its outputs remain in `scope-diagnostic.json`; it is not adopted or presented as a matched control.

## Follow-up continuation plan

The refinement preserves all six complete-guide training rows and copies validation, development, final and dialogue files byte for byte. It changes the mash and after-boil teaching answers into complete labeled stage lists, adds twenty question variants across the existing topics and adds two guide-to-follow-up contexts. There are 46 conversations about the same recipe, not additional recipes. No reserved question is copied into training.

A separate three-epoch / 138-step run initializes from the first adapter with a fresh optimizer and learning rate 0.0001. Parent weight/config/prompt hashes enter the run signature, and model identity, revision, rank/alpha/dropout and target modules must match. The original adapter and checkpoints remain untouched. Continued LoRA training explicitly enables trainable adapter weights, as required by the [PEFT API](https://huggingface.co/docs/peft/v0.13.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained); this was also verified in the installed 0.13.2 implementation.

All 65 software tests pass. The original base comparison can be reused because prompts, references, decoding, loader/evaluator and answer-masking code are unchanged. Final wording and actual-history tests are still reserved until development choices finish.

### Continuation halfway check

At step 69 / epoch 1.5, the complete guide, four-part mash answer and five-part after-boil answer all exactly match their source-checked references. The complete guide has not been lost while learning the new follow-up format. These are training-prompt checks; the previously failing fresh questions must still be generated and reviewed after the run. Parent adapter files were verified unchanged after loading.

### Completed continuation and remaining development error

The continuation finished 138 steps in 2,236.2 seconds (37.3 minutes), with 14.64 GiB peak RSS and no overlength skips. Its final full guide and both canonical multi-stage probes exactly match the source-checked references.

Under fresh development wording, the final checkpoint passes 4/5. The full guide and ingredients remain correct; both previously incomplete mash/cold-stage answers now include every requested stage with correct settings. However, the known-versus-missing answer regresses: it confuses priming quantity with temperature and falsely denies provided fermentation details. That case remains a failure.

Checkpoint 69 is being evaluated against the same complete development suite before considering another training run. The eventual selection must pass all five cases as a single model; outputs from different checkpoints will not be combined. All earlier failures and complete raw answers are preserved. No reserved final or dialogue answers have been generated yet.
