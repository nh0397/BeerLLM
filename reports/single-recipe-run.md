# One-recipe brewing tutor: measured progress

**Checkpoint under regression evaluation.** The consolidated rehearsal’s saved step 152 passes all five fitting checks: complete guide, ingredients only, timer, both missing measurements and full post-boil scope. Training is stopped at that saved point for the prior twenty-one checks. Fresh final questions remain unopened; fitting success alone is not acceptance.

The goal is one complete explanation plus direct, accurate follow-ups in ordinary language. The model must supply facts from its trained weights: no recipe lookup, source notes or recipe quantities in its system prompt. All answers are inspected against the saved source and [fixed factual rubric](../configs/helles-rubric.json). References and inspection are assistant-authored, not expert brewer-reviewed.

## Source and practical limits

[AHA's Helicious Helles](https://homebrewersassociation.org/homebrew-recipe/helicious-helles/) provides a seven-gallon / 26.5 L batch, eight ingredient entries, grain rests, mash-out, sparging, a 90-minute boil with timed additions, primary fermentation, kegging/gelatin and later lagering. Re-extraction matches saved ingredients and directions. Snapshot SHA-256: `de8d17364a4aeb1f88fa7d504e024ce05ac3a9774958ef5655237c3fe3ebd84e`.

The source omits exact mash/sparge water quantities, numeric water treatment, a separate pitching temperature, gelatin dose and carbonation/bottling instructions. A complete explanation must acknowledge those gaps, while retaining the provided primary and lagering schedules. It is not a fully specified, expert-approved brewing plan. Stable gravity readings over several days are labeled separately as a general completion check.

## Actual generated-answer results

| Model | Development, out of 5 | Original reserved wording, out of 4 | Decision |
|---|---:|---:|---|
| Matched Qwen base | 0 | 0 | Fails recipe facts |
| Complete-guide adapter, step 144 | 3 | Not run | Mash and post-boil follow-ups incomplete |
| Stage-list continuation, steps 69 and 138 | 4 each | Not run | Falsely denies a provided temperature |
| Known/unknown continuation, step 120 | 5 | 1 | Rejected on reserved wording |
| Balanced first epoch, step 150 | 1 | Not run | Resume planned second epoch |
| Balanced second epoch, step 300 | 3 | Not run | Reject; beginner explanations also 0/2 |
| Short-answer teacher, greedy | 4 | Not run | Beginner explanations 2/2; one scope failure |
| Same teacher, two beams | 5 | Not run | Explanations 1/2; reject combined 6/7 |
| Scope-contrast continuation, greedy | 5 | 3 | Explanations 1/2; reject combined 9/11 |
| Kettle/definition continuation, greedy | 5 | 4 | Explanations 2/2; select for reserved testing |

The first adapter's complete guide is correct under development wording. A generic development-only instruction to enumerate every requested stage does not repair its two incomplete follow-ups; that diagnostic prompt is not adopted.

The stage-list continuation repairs both incomplete answers but regresses on known versus absent settings. Its two checkpoints fail the same case. Answers from different checkpoints are never combined to manufacture a pass.

The known/unknown continuation passes development, then fails three reserved cases. Its new full-guide answer omits most ingredients, puts hops into a mash, invents water quantity, mislabels kettle additions as dry hopping and contradicts its own lagering statement. The timer question gets an unrelated post-boil answer. The completion question omits the stable-gravity check. Only the correction of the wrong primary temperature passes.

Selection of that failed adapter was frozen before final generation. Weight SHA-256: `02497e8be7e54e8243ebdfaa958d5a0366875f97ba84711347933d753397d502`. Complete failures and per-case reasons are preserved locally in `outputs/helles-status-eval-v1/`.

## What the numeric metrics do and do not show

The known/unknown adapter lowers development guide reference loss from 2.44468 to 0.00211 and conditional perplexity from 11.52685 to 1.00212. Four follow-ups have token-weighted loss 2.57311 → 0.09392 and perplexity 13.10657 → 1.09847.

Yet its reserved guide has conditional reference perplexity **1.00383 and an incorrect generated answer**. Teacher forcing supplies preceding reference tokens when scoring the next token. It does not establish that the model independently selects and sustains that answer. Empty lexical checks have a null pass rate; they are never presented as factual accuracy. Full-answer inspection determines acceptance.

## Training and preservation

All runs use pinned `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, float32 CPU training with four threads, rank-16 LoRA / alpha 32 / zero dropout on attention and feed-forward projections. There are 18,464,768 trainable parameters, approximately 1.182% of the combined model. This is LoRA, not QLoRA.

| Run | Conversations | Steps | Training time | Peak process RSS | Parent |
|---|---:|---:|---:|---:|---|
| Complete guide | 24 | 144 | 41.4 min | 13.80 GiB | Pinned base |
| Complete follow-ups | 46 | 138 | 37.3 min | 14.64 GiB | Complete-guide adapter |
| Known versus missing | 60 | 120 | 18.7 min | 14.93 GiB | Follow-up adapter |
| Balanced questions | 150 | 300 | ~88.6 min across both active phases | 16.02 GiB during resume | Complete-guide adapter |
| Short-answer teacher | 132 / 332 replayed | 332 | 68.3 min | 14.26 GiB | Known/unknown adapter |

The first three runs total 97.3 training minutes. Mean losses are respectively 0.398605, 0.107404 and 0.411845; changed curricula make them unsuitable as a ranking of answer quality. No examples were silently truncated or skipped. All previous datasets, adapters, checkpoints, raw answers and logs remain preserved.

Continuation loads existing LoRA weights with `is_trainable=True`, checked against the installed PEFT implementation and [official API](https://huggingface.co/docs/peft/v0.13.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained). Parent model identity, revision, LoRA settings and weight/config/prompt hashes enter the run signature. Each continuation has a fresh optimizer; interrupted runs resume their own saved optimizer.

## Balanced question coverage and remaining gates

The balanced curriculum has 18 full-guide requests, 12 canonical follow-ups, 76 direct variants, 30 conversational examples and 14 known/unknown contrasts. It covers every stage, short questions without the recipe name and follow-ups after a full guide. All complete-guide targets retain the original source facts. There are 22,741 supervised answer tokens per pass; the longest conversation is 971 tokens. All 150 examples fit.

This run starts from the first complete-guide adapter at learning rate 0.00005, saving each epoch. The first epoch saved at step 150 after approximately 44.4 minutes and was deliberately stopped before epoch-end validation. No final training metrics or validation loss are claimed for that interrupted phase. The generated complete guide passes, but four development follow-ups fail: ingredients and mash receive unrelated water/settings answers, cold stages omit cooling/pitching/primary, and known/unknown settings falsely deny the primary schedule. The planned second epoch resumes the same run, data, optimizer and scheduler from checkpoint 150 using `--resume`. The first checkpoint and its failed answers remain preserved.

The original final set is consumed and now becomes an additional regression gate. Four new confirmation phrasings were frozen before balanced training; the original four-turn actual-history dialogue remains reserved. No evaluation question is copied verbatim into training. Acceptance also includes two direct explanation-development questions under different wording, so the model must demonstrate basic understanding before reserved chat generation. Acceptance requires all five original development questions, both explanation-development questions, all four regression questions, all four fresh confirmation questions and all four original dialogue turns plus two fresh beginner-explanation turns to pass as one model. This tightens coverage without relaxing any factual requirement.

The six-turn natural dialogue retains the original four questions in order, then asks for plain meanings of wort/pitching and the reason for cleaning and sanitizing. It was frozen before checkpoint selection. The dialogue feeds each model its own generated replies. Reference answers are saved for review and never inserted into history. Both confirmation and dialogue concern the one trained recipe, not unseen recipe families. Greedy decoding, model pin, system prompt, loader/evaluator and suite hashes must match between before/after runs.

All 71 software tests pass, including frozen inputs, question separation, source restrictions, continuation signatures and interrupted dialogue resumption without reference leakage. Data and weights remain local and ignored by Git. See [the reproduction commands](../docs/SINGLE-RECIPE.md). A successful bounded demonstration would still need broader testing and expert review before expansion.


## Completed balanced run and focused correction

The balanced run completes step 300. It preserves the complete guide and repairs the ingredient and mash answers, scoring 3/5 development. The post-boil question still receives only kegging/gelatin/lagering. The known/unknown answer invents a four-gallon batch and packaging instructions and treats the provided batch size as absent. Both explanation questions are off-topic: wort/pitching gets a boil answer, and cleaning/sanitizing gets a lagering answer. This model is rejected before any fresh confirmation or natural-dialogue answers are generated.

The first phase reports 44:24 of progress time. The resumed phase records 2,654.6 project-monotonic seconds (44.2 minutes) but 60,523.3 Trainer elapsed seconds (16.8 hours), with a long overnight progress gap. Do not describe the latter as continuous compute. Resume-phase mean loss 0.066783 is not a whole uninterrupted-run comparison. Recorded peak RSS during resume is 16.02 GiB. All 150 rows fit.

The next run returns to the known/unknown adapter, which already passed the original five development questions, rather than inheriting the rejected balanced weights. Its 132 examples include ten full guides, twelve canonical follow-ups, 76 recipe follow-up variants, twelve plain-language concept variants, fourteen known/unknown contrasts and eight conversations. One hundred short examples are replayed three times: 332 optimizer steps, not 332 distinct recipes or questions. Cold-stage and completion answers start with the requested action; known settings and real gaps are separated clearly. General explanations define wort/pitching and distinguish cleaning from sanitizing, including two follow-ups after a complete guide.

This is standard assistant-only LoRA training with the existing replay sampler, learning rate 0.0001 and one pass over the weighted dataset. No custom loss, factual system prompt or lookup is introduced. All full-guide source facts remain intact; evaluation files are copied unchanged. All 132 examples / 332 replayed rows fit, with 32,047 supervised answer tokens and maximum 880 tokens. The original five development questions, two explanation-development questions and consumed four-case regression set must pass before selecting the model. The four confirmation questions and six-turn actual-history conversation are still unopened.


## Focused teacher: explanations work; decoding scope remains

The teacher finishes all 332 steps with no skips. It passes the complete guide, ingredients, mash and known/unknown questions, including the correct seven-gallon batch and absent packaging instructions. Both beginner explanations pass: wort/pitching are defined in everyday language, and cleaning is distinguished from sanitizing. The broad post-boil question still receives only later kegging/gelatin/lagering. This is 6/7 development checks, not acceptance.

Two uniform LoRA-strength diagnostics do not fix it. Alpha 16 (half the trained strength) invents extra carbonation, cabinet, bottling and re-kegging operations. Alpha 40 (1.25 times trained strength) returns the same incomplete later-stage answer. Both are rejected; the original alpha-32 weights/config remain unchanged.

A diagnostic with **two deterministic beams and length penalty 1.0** produces all five requested post-boil stages, with correct primary/lagering settings and genuine gaps. It considers alternative continuations instead of making only locally greedy word choices. This is one development result, not a final score. A packaged candidate preserves the exact trained weights/config and system prompt, adding only an explicit decoding policy to its chat metadata. Weight SHA-256 remains `9909725a0415b15e5fa088f1d65778995710642ae1fd2dd89e632d4953198c9a`.

The final protocol must use the same beam policy for both the pinned base and adapter, with fresh before/after files and matching evaluator/loader signatures. Earlier greedy baselines cannot be reused as beam controls. All seven development checks and four consumed regression cases must pass before selecting this candidate; four fresh confirmation questions and the six-turn actual-history conversation remain unopened. No source facts are inserted into prompts or postprocessed into answers. All 71 software tests pass, covering policy propagation, unchanged greedy defaults, package integrity and policy-aware dialogue resumption.


## Beam search exposes a trade-off rather than completing the goal

The packaged two-beam candidate passes all five original development cases. Its full guide remains source-faithful, and the broad post-boil answer includes cooling, yeast addition, primary fermentation, kegging/gelatin and later lagering. However, it answers the cleaning/sanitizing question with that unrelated post-boil list. The definitions of wort/pitching remain correct. Both greedy and two-beam configurations therefore pass 6/7 development cases, with different failures. Their correct answers are not combined into an artificial pass.

The observed greedy cleaning answer has total sequence NLL 6.57217 over 38 tokens, whereas the unrelated 134-token process list has total NLL 1.97831. For the two observed cold-stage candidates, the complete answer needs length penalty above approximately 0.448 to outrank the incomplete answer. The cleaning pair would need a penalty below approximately -0.953 to favor its correct answer. No single length penalty can rank both of these observed pairs correctly. Re-tokenized lengths match their saved generation lengths. This diagnostic ranks these particular candidates, not all possible outputs or decoding methods.

Two-beam generation is also slow on this capped Intel CPU: the complete guide takes 1,019.3 seconds (17.0 minutes), versus 371.3 seconds (6.2 minutes) for the greedy teacher. The candidate is rejected before opening confirmation or actual-history tests. All new-policy outputs and diagnostics remain local under `outputs/helles-beam-eval-v1/`. A final comparison still requires matching controls under the final selected policy and current evaluator/loader; historical greedy outputs are not silently relabeled.

## Teach requested scope directly

The correction initializes from the greedy teacher, preserving its correct guide, ingredients, mash, known/unknown settings and two explanations. Eight minimal-pair sets distinguish everything after boiling from only what follows primary fermentation. They retain similar wording while changing the requested starting point. Full-range answers explicitly include the early operations and primary schedule; tail-only answers give kegging/gelatin and later lagering. A short, complete cold-stage answer begins with the requested scope, “After the boil,” rather than changing every target to a new generic opening.

There are 62 teaching conversations about the same source recipe: six full guides, twelve canonical follow-ups, twelve direct variants, sixteen scope-pair examples, four concept variants, eight conversations and four known/unknown contrasts. Replaying the 44 short examples three times produces 150 optimizer steps. All rows fit, with 14,763 supervised answer tokens and maximum 880 tokens. One standard assistant-only LoRA epoch runs at learning rate 0.0001. No new model, custom loss, factual prompt or runtime question router is introduced. All 71 software tests pass, including scope-pair labels and reserved-question separation. Confirmation and natural-dialogue outputs remain unopened.

## Scope correction results and next correction

The 150-step scope run finishes in 40.8 minutes at peak process RSS 14.65 GiB, mean loss 0.063477, with zero skipped rows. It passes 5/5 recipe development, 1/2 explanation development and 3/4 consumed regression checks. Its glossary correctly defines both words, then adds the confusing claim “This is a separate exact pitching temperature from the general brewing setup.” Its boil-timer reply incorrectly supplies the post-boil process. These whole answers fail; neither is trimmed to create a pass.

Checkpoint 75 passes the glossary and broad post-boil screen, but its timer answer invents torrified wheat in cooled wort and repeats until the token limit. It is rejected without further full-suite evaluation. Both checkpoint screens and all generated answers are retained.

The next correction continues from step 150 with balanced practice on in-boil additions, the post-boil sequence and concise word definitions. It retains all six original full-guide targets, twelve canonical follow-ups and six original conversations. New wording is distinct from evaluation questions. The learning rate is halved to 0.00005 and short-answer replay is disabled for this small pass. No recipe facts are added to the system prompt, and no runtime routing or answer editing is introduced. Fresh confirmation and natural dialogue remain unopened.

The kettle/definition correction completes 54 steps in 860.9 seconds (14.3 minutes), peak RSS 15.72 GiB, mean training loss 0.045923, with zero skipped examples. Both initial development screens pass: a clean three-sentence glossary and the correct 90-minute boil with measured additions at 30/80 minutes elapsed. Full selection and reserved testing are still pending; the screen alone does not establish acceptance.

## One candidate passes selection

The final kettle/definition adapter passes all eleven full-answer checks under one fixed greedy policy. Both differently worded complete guides include every ingredient and the whole procedure. The kettle-timer answer, primary-versus-lagering correction, general completion check, known/unknown distinction and two beginner explanations all pass. No output is trimmed or combined with another checkpoint.

Weight SHA-256: `b5088093f98cf0fbb78d5b6aea646edddde13cea077c19f297d59a6a6ef07299`. Selection is frozen locally in `outputs/helles-kettle-eval-v1/selection.json`, including the policy, evaluator/loader, rubric, suite and profile hashes, before reserved generation. The deployment package contains unchanged trained weights and an explicit greedy policy. A fresh baseline with the current evaluator is still required; older controls are historical and will not be relabeled as this comparison.

The frozen candidate passes all four fresh confirmation questions on its first reserved evaluation: a complete guide, measured kettle additions, correction of a wrong fermentation temperature and an evidence-based completion check. The six-turn actual-history conversation is next. These are new phrasings for the same trained recipe; they do not establish unseen-recipe performance.

## Actual history exposes one remaining omission

The first six-turn conversation passes turns 1, 2, 3, 5 and 6. Turn 4 asks for both exact water quantities and gelatin dose. Its answer describes absent water quantities and the seven-gallon batch, but never answers the gelatin part. This is an incomplete answer and fails; the candidate is not accepted despite 15/15 non-conversational checks. The dialogue is now consumed regression evidence. A general multipart-answer instruction is being screened without inserting recipe facts or changing trained weights.

The generic multipart-answer instruction also fails the consumed conversation question: the response again omits gelatin and adds unrelated packaging guidance. That prompt is rejected and never adopted. The next continuation retains the original system prompt and starts from the kettle adapter. Its 53 rows include the six original guides, twelve canonical follow-ups, six original conversations, kettle and explanation practice, compound missing-quantity answers, mixed known/unknown questions and four conversational variants. Learning rate is 0.00003 with no replay, one epoch and checkpoints at step 30 and the end. All examples fit; 7,394 answer tokens are supervised, maximum input 898 tokens.

The first four confirmation questions and six-turn conversation are now consumed regression evidence. Before this new training, a second four-question set and a fresh six-turn conversation are frozen under different wording. Selection must retain the existing fifteen non-conversational answers and original six-turn conversation, then pass all ten fresh checks. The factual rubric is unchanged. The final before/after comparison will use the fresh ten questions and the same prompt/policy for base and adapter.

The 53-step compound pass completes in 473.0 seconds (7.9 minutes), peak RSS 15.62 GiB, mean loss 0.398122, with zero skips. It still fails the consumed conversation question: it names an absent mineral profile and gelatin dose but does not clearly answer the requested water amount, then adds an incoherent calendar/settings sentence. A training-fit diagnostic also fails both an exact training question and the consumed question without history, answering water only. This shows the new combined answer is not fitted yet; the error is not limited to long history. No final tests are opened. A two-epoch fitting continuation uses the same frozen 53 examples, learning rate 0.0001 and exact-training generation probes at each saved epoch.

At the first saved fitting checkpoint (step 53), the exact training probe produces the complete water-and-gelatin target, including both real omissions and the distinction between batch size and starting water. Training is deliberately interrupted after that fully saved checkpoint to evaluate before more updates. Its optimizer and run configuration remain resumable; no final two-epoch metrics are claimed. This is a fitting check, not reserved-test success.

## Preserve complete-guide intent while answering gaps

The fitted compound checkpoint passes the original seven development cases and the saved-history compound screen. It fails the consumed complete brew-sheet request: after “Let’s make Helicious Helles,” it supplies only missing pitching settings and primary/lagering schedules, omitting ingredients and most stages. Evaluation is stopped after that known failure; no new final questions are generated. A fixed 50% interpolation of the earlier kettle adapter and this checkpoint also fails the compound screen and the guide-prefix screen. It is rejected; 25% and 75% candidates are saved but not evaluated. This is constant LoRA-parameter interpolation, not query routing or output combination, and no blend is adopted.

The next pass retains all 53 compound teaching conversations and adds twelve whole-recipe requests that also mention missing details. Their answer is the unchanged, complete source guide. This directly distinguishes a full recipe including gaps from a request for gaps alone. It continues from the exact saved step-53 weights copied into a metadata-equipped staging adapter; SHA verification confirms unchanged checkpoint weights/config. The original system prompt and all factual criteria remain unchanged. The second confirmation set and second conversation remain unopened.

The full-guide scope rehearsal completes 65 steps in 956.1 seconds (15.9 minutes), peak RSS 16.25 GiB, mean training loss 0.048229, with zero skipped examples. It supervises 15,518 answer tokens; maximum input is 898 tokens. A short intent screen precedes full regression and reserved testing.

The final guide-scope adapter repairs the consumed complete-guide request and preserves the other completed stage/explanation screens, but the ingredient-only question now receives a full guide and hits its 400-token budget. It is rejected, and the full regression run is stopped early. A halfway checkpoint is screened on the complete-guide prefix, the full ingredient answer and the saved-history compound request before any further training. The second reserved confirmation and conversation are still unopened.

## Fresh attention-only adapter instead of further patching

The guide-scope halfway checkpoint also returns a truncated full guide to the ingredient-only request. Adding a generic ingredient-only stopping rule to the final checkpoint does not repair it. That rule is rejected; the original neutral teaching prompt remains in use. No new recipe facts are injected and no runtime query classifier is introduced.

The next run starts from the unmodified pinned Qwen base, with rank-16 / alpha-32 LoRA on query and value projections only, dropout 0.05, learning rate 0.0001, two planned epochs and no replay. It inherits no previous adapter. This reduces the trainable parameter count from 18,464,768 to 2,179,072 (about 0.141% of the combined model). Whether that preserves instruction-following better is an empirical question, not an acceptance claim.

Its 165 distinct teaching conversations combine broad follow-up coverage, 22 full guides, 12 canonical follow-ups, 106 variants, 14 known/unknown contrasts and 11 conversation examples. All fit: 25,168 supervised answer tokens, maximum input 886 tokens. There are 330 planned optimizer steps, with checkpoints and four exact-training generation probes each epoch: guide, ingredients, timer and combined quantities. Previous fifteen non-conversational cases and six-turn chat remain regression gates. The untouched second confirmation/chat wording stays reserved; one newly frozen ingredient-only wording brings the final check to five independent questions plus six actual-history turns. All use the original system prompt, and exact question overlap with training is checked.

## Feed-forward adapter candidate

The attention-only run completes 330 steps in 2,534.1 seconds (42.2 minutes), peak RSS 15.15 GiB, mean training loss 1.597382, with zero skipped examples. Reference validation loss falls from 1.42065 to 0.94326, but both epochs fail all four exact-training generation checks. They invent quantities, additions and gelatin settings, or deny the available procedure. Neither checkpoint is accepted; no final questions are generated.

A diagnostic keeps the trained feed-forward (`gate_proj`, `up_proj`, `down_proj`) LoRA factors from the guide-scope model and removes its attention adapters. It retains 14,106,624 parameters; these were trained as part of the earlier 18,464,768-parameter adapter, not as a fresh feed-forward-only run. The pinned base, alpha, rank, original system prompt and greedy policy remain the same. All queries use this one fixed module subset. There is no answer lookup, per-question weight choice or factual prompt.

The candidate passes the complete ingredient-list screen and the saved-history water/gelatin question. Its truncated guide-prefix diagnostic starts correctly but is not counted as a full-guide pass. All fifteen earlier non-conversational questions and the original six-turn actual-history chat are now being generated in full before selection. `scripts/build_helles_mlp.py` reproduces the subset with factor/rank checks, immutable-source hashes and a new output directory. Rebuilding produces byte-identical weights/config and the same system prompt as the diagnostic candidate. Previous models and all failed outputs remain local.

## Consolidate all response types together

The feed-forward-only candidate passes the complete guide and ingredient list but fails the broad post-boil question by omitting cooling, yeast and primary fermentation. Adding query/key adapters does not repair that range. Using the earlier kettle attention factors restores the post-boil answer but returns a truncated full guide for ingredients only. Half-strength earlier attention has the same ingredient failure. Short prefix screens at strengths 0.125, 0.25 and 0.375 also do not preserve all four intended response types. These are rejected diagnostics, never combined into a pass. No final questions are opened.

The focused 65-row repair set contained 18 full guides but only one ingredient-only target. The consolidated rehearsal returns to the kettle adapter, which had passed all fifteen non-conversational cases and five of six conversation turns. It uses broad coverage from the 165-row curriculum plus ten new ingredient-only requests: 175 distinct conversations. Replaying short factual/variant rows twice gives 303 presentations, including 34 ingredient-only targets versus 22 full guides. The combined missing-quantity answers, complete post-boil sequence and all other question types remain in the same run. There are 37,130 supervised answer tokens and no context skips; the longest input is 886 tokens.

This is the existing standard assistant-only LoRA objective, all seven projections, rank 16 / alpha 32 / zero dropout, learning rate 0.0001, one weighted pass. No custom loss, new serving rule or module blend is adopted. Five exact-training generation probes check guide, ingredients, timer, compound quantities and complete post-boil scope. A new complete-post-boil wording is frozen alongside the unused final questions, bringing final evaluation to six single-turn questions plus six actual-history turns. The previous twenty-one regression checks and original factual criteria still apply.

At saved step 152, all five exact-training generations pass whole-answer inspection. The guide retains all facts and stages, ingredients stop after the list, timing uses the correct 90-minute countdown, both quantity omissions are explicit, and the post-boil answer includes cooling/yeast/primary/kegging/lagering. Training is deliberately interrupted after that checkpoint for regression evaluation. The optimizer/config remain resumable; no full-303-step metrics are claimed. A staged adapter copies the exact saved weights/config, matching tokenizer/system metadata and the actual v13 training-data hash. No rejected projection subset, weight blend or prompt cue is used by this candidate.
