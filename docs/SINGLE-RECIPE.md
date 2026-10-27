# One complete recipe before scaling

This experiment teaches only **Helicious Helles**, using the saved AHA source. The target is a complete, understandable recipe plus accurate follow-up answers. It starts from the pinned Qwen2.5-1.5B base, not the previous multi-recipe adapter. Recipe facts are not embedded in the system prompt or silently fetched during evaluation.

The lesson preserves the source's seven-gallon batch, every ingredient, grain rests, hop countdown, primary fermentation, kegging/gelatin and later lagering. Water quantities/profile, a separate pitching temperature, gelatin dose and carbonation instructions are genuinely absent; the model must identify those gaps while retaining the temperature settings that are present. This is a complete explanation of the available source, not an expert-approved brew sheet with every equipment-dependent parameter settled.

The dataset contains 24 conversations: six full-guide requests, twelve direct follow-ups and six conversations. Six epochs give 144 optimizer steps and 36 supervised full-guide presentations. The validation guide repeats the same recipe facts under different wording. None of these counts represent independent recipes or general brewing accuracy.

## Prepare and record the baseline

```bash
.venv/bin/python -m beerllm.recipe_focus prepare

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-focus.json --data-dir data/helles_focus_v1 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v1/development.jsonl \
  --mode base --output outputs/helles-focus-eval-v1/development-before.json
```

## Train and inspect the whole guide

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-focus.json --data-dir data/helles_focus_v1 \
  --eval-limit 1 --probe-suite data/helles_focus_v1/probes.jsonl \
  > outputs-helles-train.log 2>&1
```

The two checkpoints generate an entire training guide and a hop-timer answer. Review the actual answers against [the frozen rubric](../configs/helles-rubric.json). Passing token-pattern checks or low reference loss cannot establish success. Resume an interrupted run using the identical command plus `--resume`, appending to the log. Preserve the existing output; use a new directory for a different experiment. Run only one model job at a time.

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v1/development.jsonl \
  --mode adapter --adapter outputs/helles-focus-v1/adapter \
  --output outputs/helles-focus-eval-v1/development-after.json

.venv/bin/python -m beerllm.benchmark compare --suite data/helles_focus_v1/development.jsonl \
  --before outputs/helles-focus-eval-v1/development-before.json \
  --after outputs/helles-focus-eval-v1/development-after.json \
  --output outputs/helles-focus-eval-v1/development-comparison.json
```

## Repair incomplete follow-ups without restarting the recipe

The first adapter learns the complete guide but scores 3/5 on development because two multi-part questions receive partial answers. The continuation adds varied questions and complete stage lists while preserving the guide and all evaluation bytes. It starts from the original adapter with a fresh optimizer at a lower learning rate. Parent adapter hashes and compatible LoRA settings are checked before training/resume.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-followups.json --data-dir data/helles_focus_v2 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-followups.json --data-dir data/helles_focus_v2 \
  --eval-limit 1 --probe-suite data/helles_focus_v2/probes.jsonl \
  > outputs-helles-followups-train.log 2>&1

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v2/development.jsonl \
  --mode adapter --adapter outputs/helles-followups-v1/adapter \
  --output outputs/helles-followup-eval-v1/development-after.json

.venv/bin/python -m beerllm.benchmark compare --suite data/helles_focus_v2/development.jsonl \
  --before outputs/helles-focus-eval-v1/development-before.json \
  --after outputs/helles-followup-eval-v1/development-after.json \
  --output outputs/helles-followup-eval-v1/development-comparison.json
```

The continuation contains 46 conversations and runs 138 steps. Generated probes check the full guide and both multi-stage follow-ups. The first run, source snapshots and evaluation files are preserved.

## Distinguish known temperatures from missing quantities

The stage-list continuation scores 4/5: it fixes both multi-stage answers, but its two checkpoints falsely deny a supplied temperature when asked about mixed known/unknown settings. The next refinement adds fourteen contrast examples while retaining the working guide and stage questions. Evaluation files remain byte-identical to the original suite.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --config configs/helles-status-refinements.json --output data/helles_focus_v3

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-status.json --data-dir data/helles_focus_v3 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-status.json --data-dir data/helles_focus_v3 --eval-limit 1 \
  > outputs-helles-status-train.log 2>&1

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v3/development.jsonl \
  --mode adapter --adapter outputs/helles-status-v1/adapter \
  --output outputs/helles-status-eval-v1/development-after.json

.venv/bin/python -m beerllm.benchmark compare --suite data/helles_focus_v3/development.jsonl \
  --before outputs/helles-focus-eval-v1/development-before.json \
  --after outputs/helles-status-eval-v1/development-after.json \
  --output outputs/helles-status-eval-v1/development-comparison.json
```

This continuation initializes from `outputs/helles-followups-v1/adapter`, saves in a new directory and trains for two passes over 60 conversations. Full-guide and stage-list rehearsal remains included. The source recipe and old models are preserved.

## Fresh wording and actual conversation

Only after development choices finish, repeat the benchmark pair and comparison with `data/helles_focus_v3/final.jsonl`, the selected continuation adapter and `outputs/helles-status-eval-v1/final-{before,after,comparison}.json`. These are different questions about the same trained recipe, not new recipe families.

The dialogue test feeds each model its own generated answer before asking the next question. Reference answers are saved for review, never inserted into the live history. It can resume an interrupted evaluation with the same output signature.

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.recipe_focus dialogue \
  --protocol data/helles_focus_v3/dialogue.json \
  --output outputs/helles-status-eval-v1/dialogue-before.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.recipe_focus dialogue \
  --protocol data/helles_focus_v3/dialogue.json \
  --adapter outputs/helles-status-v1/adapter \
  --output outputs/helles-status-eval-v1/dialogue-after.json
```

The original gate required all five development cases, four final wording cases and four dialogue turns. After the final-wording failure, those four consumed cases become regression tests and four fresh confirmation cases are added; all must pass source-based full-answer inspection. It establishes a bounded one-recipe demonstration only. References and inspection remain assistant-authored, not independent expert validation. Results belong in [the run report](../reports/single-recipe-run.md).

The following command reproduces the rejected status candidate for diagnosis; it is not the accepted chat model. A passing replacement has not been selected yet:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.chat --adapter outputs/helles-status-v1/adapter
```

Ask for the complete Helicious Helles recipe, then ask follow-ups. Use `exit` to stop. The adapter saves its focused system prompt for chat. Old adapters and all collected recipes remain preserved.


## Balanced coverage after a failed reserved test

The status continuation was rejected: development 5/5, reserved wording 1/4. The new run broadens all topics and conversation contexts rather than treating that score as success. It initializes from the first complete-guide adapter and preserves every earlier experiment.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --config configs/helles-balanced-refinements.json --output data/helles_focus_v5
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-balanced.json --data-dir data/helles_focus_v5 --eval-limit 1 \
  > outputs-helles-balanced-train.log 2>&1
```

The original `final.jsonl` is now a regression suite used during selection. `confirmation.jsonl` contains four newly reserved phrasings, frozen before this training run. The original actual-history `dialogue.json` remains reserved. Do not use either reserved set to select a checkpoint. Existing local datasets are frozen; skip preparation when the recorded dataset is already present, and use a new output directory for changed builder/config inputs.


For the selected balanced checkpoint, use `--protocol configs/helles-natural-dialogue.json` for both the base and adapter dialogue runs. It retains the original four turns, then adds two beginner explanation questions. Freeze selection before generating either reserved confirmation or natural-dialogue answers. The test must use each model's own preceding replies.


## Teach direct answers and beginner concepts

The balanced model ends at 3/5 original development and 0/2 explanation cases. This correction returns to the earlier five-case development winner, adds explicit beginner explanations and increases short-answer rehearsal. Ten full-guide rows retain the recipe. All old outputs stay preserved.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --config configs/helles-teacher-refinements.json --output data/helles_focus_v7
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-teacher.json --data-dir data/helles_focus_v7 --check-data
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-teacher.json --data-dir data/helles_focus_v7 --eval-limit 1 \
  > outputs-helles-teacher-train.log 2>&1
```

There are 132 distinct teaching conversations about one source recipe. Replaying the 100 short follow-ups three times produces 332 training rows. `--resume` uses the identical config/data/parent hashes after an interruption. Evaluate with the unchanged development and consumed regression suites plus `configs/helles-explanation-dev.jsonl`; only after selecting a passing candidate open `confirmation.jsonl` and `configs/helles-natural-dialogue.json`.


## Explicit decoding policy and immutable chat package

The greedy teacher scores 6/7 development checks. A two-beam diagnostic repairs its remaining scope failure. This candidate needs a complete matched evaluation before it is recommended.

```bash
.venv/bin/python -m beerllm.recipe_focus package \
  --adapter outputs/helles-teacher-v1/adapter \
  --output outputs/helles-teacher-beam-v1/adapter \
  --decoding-config configs/helles-decoding.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v7/development.jsonl \
  --mode base --decoding-config configs/helles-decoding.json \
  --output outputs/helles-beam-eval-v1/development-before.json

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.benchmark run \
  --model-id Qwen/Qwen2.5-1.5B-Instruct --suite data/helles_focus_v7/development.jsonl \
  --mode adapter --adapter outputs/helles-teacher-beam-v1/adapter \
  --decoding-config configs/helles-decoding.json \
  --output outputs/helles-beam-eval-v1/development-after.json
```

Use the same decoding file for the explanation, regression and confirmation benchmark pairs, and for both `recipe_focus dialogue` runs. Comparisons reject mismatched policies or evaluator/loader versions. The package copies trained weights/config unchanged, adds the policy to `beerllm.json` for normal chat and records source/output hashes. It refuses changed or conflicting packages. Omitting the policy retains the original neutral greedy behavior for historical experiments. The chat default is promoted only after all quality gates pass.


## Distinguish whole sequences from later-only questions

The beam candidate fixes the recipe-stage case but regresses the cleaning explanation. It is rejected. The next greedy correction teaches paired starting points while rehearsing the six working development answers.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --config configs/helles-scope-refinements.json --output data/helles_focus_v8
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-scope.json --data-dir data/helles_focus_v8 --check-data
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-scope.json --data-dir data/helles_focus_v8 --eval-limit 1 \
  > outputs-helles-scope-train.log 2>&1
```

Use `configs/helles-greedy-decoding.json` for both base/adapter benchmark and dialogue runs if this candidate qualifies. A final current-version base control is still required. The old beam package and greedy/beam outputs remain preserved; none is promoted as the chat default until all factual and conversation gates pass.

## Separate kettle timing from post-boil instructions

The scope run passes 9/11 selection checks. Checkpoint 75 still fails the timer, so neither checkpoint is accepted. The next pass keeps the six original guides, twelve canonical follow-ups and six original conversations, adds kettle-timing and plain-definition variants, and continues from the final scope adapter. There are 54 examples with no replay; one epoch uses learning rate 0.00005.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --source data/helles_focus_v1 --config configs/helles-kettle-refinements.json \
  --output data/helles_focus_v9

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-kettle.json --data-dir data/helles_focus_v9 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-kettle.json --data-dir data/helles_focus_v9 --eval-limit 1 \
  > outputs-helles-kettle-train.log 2>&1
```

Use the already-frozen local datasets when present. Do not rerun earlier failed experiments simply to use a saved adapter. All original checkpoints and results remain available; a different run needs a new output directory.

## Teach compound requests without hiding the failed conversation

The kettle adapter passes fifteen non-conversational questions and five of six actual-history turns. Its reply to a combined water/gelatin question omits gelatin. A generic prompt cue does not fix it. The next pass keeps the original prompt and teaches both single-item and combined requests, including known/unknown contrasts.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --source data/helles_focus_v1 --config configs/helles-compound-refinements.json \
  --output data/helles_focus_v10

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-compound.json --data-dir data/helles_focus_v10 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-compound.json --data-dir data/helles_focus_v10 --eval-limit 1 \
  > outputs-helles-compound-train.log 2>&1
```

The 53 rows all fit, no short-fact replay is applied, and the learning rate is 0.00003. The prior fifteen non-conversational questions and first six-turn chat become regression checks. A second four-question confirmation file and `configs/helles-confirmation-dialogue.json` are frozen before training and remain unopened until selection. Neither references nor recipe facts are inserted into inference history.

## Verify that the new answer actually fits

One low-rate compound pass still answers water only, even for an exact training question. A second fitting pass keeps the same data, saves each epoch and generates an exact training answer at each save.

Export its training-only probe first:

```bash
.venv/bin/python - <<'PROBE'
from pathlib import Path
from beerllm.common import read_jsonl, write_jsonl
rows = read_jsonl('data/helles_focus_v10/train.jsonl')
row = next(r for r in rows if r['messages'][-2]['content'] ==
           'What are the exact mash-water volume and gelatin quantity?')
write_jsonl(Path('outputs/helles-compound-eval-v1/training-probe.jsonl'), [{
    'case_id': 'exact-compound-training-fit', 'category': 'training_fit',
    'source_url': 'https://homebrewersassociation.org/homebrew-recipe/helicious-helles/',
    'messages': row['messages'], 'checks': [], 'max_new_tokens': 180}])
PROBE
```

```bash
HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-compound-fit.json --data-dir data/helles_focus_v10 \
  --eval-limit 1 --probe-suite outputs/helles-compound-eval-v1/training-probe.jsonl \
  > outputs-helles-compound-fit-train.log 2>&1
```

The local probe is the `What are the exact mash-water volume and gelatin quantity?` training row, exported in benchmark form with category `training_fit`, no lexical checks and a 180-token response budget. Its reference remains outside the generation prompt. The first saved checkpoint, step 53, generates the full combined answer and then passes the original failed question with its saved conversation history. The run is intentionally interrupted after that checkpoint for development checks; its optimizer remains resumable with the identical arguments plus `--resume`. This is not a completed two-epoch run. Do not use `--max-steps 53` as a substitute: that changes the learning-rate schedule. Earlier questions and the full original conversation still need to pass before final selection.

## Rehearse whole recipes that also ask about gaps

The fitted compound checkpoint still fails one consumed full-guide request, responding only about missing settings. A fixed 50% parameter blend does not resolve both errors and is rejected. The next pass retains all 53 compound examples and adds twelve full-guide questions that also request source omissions. Their answer remains the unchanged complete guide.

The initial adapter at `outputs/helles-compound-step53-v1/adapter` is a copy of checkpoint 53 with matching tokenizer/profile metadata from its same-data parent. Its weight and config hashes match the original checkpoint; the original files are preserved.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --source data/helles_focus_v1 --config configs/helles-guide-scope-refinements.json \
  --output data/helles_focus_v11

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-guide-scope.json --data-dir data/helles_focus_v11 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-guide-scope.json --data-dir data/helles_focus_v11 --eval-limit 1 \
  > outputs-helles-guide-scope-train.log 2>&1
```

## Fresh attention-only LoRA

Successive all-projection continuations trade scope failures. The next experiment starts from the unmodified Qwen base and trains only query/value LoRA layers, using the combined teaching coverage. It inherits no previous adapter.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --source data/helles_focus_v1 --config configs/helles-attention-refinements.json \
  --output data/helles_focus_v12

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-attention.json --data-dir data/helles_focus_v12 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-attention.json --data-dir data/helles_focus_v12 --eval-limit 1 \
  --probe-suite outputs/helles-attention-eval-v1/training-probes.jsonl \
  > outputs-helles-attention-train.log 2>&1
```

The probe file contains four exact training rows exported in benchmark form (`training_fit`, empty lexical checks): complete-guide question, canonical ingredients question, canonical boil-timer question and `What are the exact mash-water volume and gelatin quantity?`. Budgets are 1100, 400, 180 and 180 tokens. It is a fitting diagnostic, never a final test. Use the same arguments plus `--resume` after an interruption. Run one model process at a time.

## Reproduce the feed-forward candidate

The fresh attention-only run fails all four training-generation checks at both epochs. The next candidate keeps the trained feed-forward LoRA modules from `helles-guide-scope-v1` and removes attention adapters. This is one fixed adapter for every query. Its 14.1 million retained parameters were learned during the earlier all-projection training run.

```bash
.venv/bin/python scripts/build_helles_mlp.py

.venv/bin/python -m beerllm.recipe_focus package \
  --adapter outputs/helles-mlp-selected-v1/adapter \
  --output outputs/helles-demo-v1/adapter \
  --decoding-config configs/helles-greedy-decoding.json
```

The builder preserves its source and refuses an existing output. It verifies the factor pairs, rank, target modules, parameter count and matching model/profile. The rebuilt weights and config match the screened candidate byte for byte; the system prompt is unchanged. Passing a guide-prefix probe is not a complete-guide pass. All original factual requirements, previous regression questions and fresh checks remain required before calling the candidate accepted.

## Consolidated rehearsal from the earlier passing adapter

Module subsets trade cold-stage and ingredient-only scope failures. The next run uses the original standard trainer and the earlier kettle adapter, which had passed all fifteen non-conversational questions. It practices all teaching coverage together, adding ten ingredient-only variants. Short-answer replay gives 303 presentations of 175 distinct conversations, with 34 ingredient targets versus 22 guides.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --source data/helles_focus_v1 --config configs/helles-rehearsal-refinements.json \
  --output data/helles_focus_v13

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-rehearsal.json --data-dir data/helles_focus_v13 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-rehearsal.json --data-dir data/helles_focus_v13 --eval-limit 1 \
  --probe-suite outputs/helles-rehearsal-eval-v1/training-probes.jsonl \
  > outputs-helles-rehearsal-train.log 2>&1
```

Its training-only probes cover the same four exact questions as the attention-only run plus the canonical complete cold-stage question, with a 220-token response budget. Checkpoints/probes are preserved. The six fresh single-turn questions and six fresh conversation turns stay unopened until the previous regression cases pass. No rejected module subset or serving-prompt cue is promoted.

## Preserve previous examples when adding a new distinction

The completed rehearsal passes fourteen of fifteen independent questions and all six actual-history turns. The remaining standalone timer question confuses ongoing boiling with the after-boil stage. Several earlier successful kettle phrasings were accidentally omitted when manually combining curricula.

The refiner now retains previous teaching datasets explicitly. List older parents first; later parents win for identical full inputs. Distinct histories remain distinct. Parent file hashes, the shared frozen recipe, system prompt and original held-question guard are checked.

```bash
.venv/bin/python -m beerllm.recipe_focus refine-followups \
  --source data/helles_focus_v1 \
  --retain-training-from data/helles_focus_v9 data/helles_focus_v13 \
  --config configs/helles-boil-state-refinements.json --output data/helles_focus_v14

HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.train \
  --config configs/qwen-helles-boil-state.json --data-dir data/helles_focus_v14 --check-data

HF_HUB_OFFLINE=1 caffeinate -i .venv/bin/python -u -m beerllm.train \
  --config configs/qwen-helles-boil-state.json --data-dir data/helles_focus_v14 --eval-limit 1 \
  --probe-suite outputs/helles-boil-state-eval-v1/development-probe.jsonl \
  > outputs-helles-boil-state-train.log 2>&1
```

The probe is the consumed original timer question, reclassified as `seen_fact_new_wording` development evidence. Its reference stays outside the generation prompt and it is not added to training. The twelve fresh final questions remain separate. Retention adds no runtime lookup or routing.
