# One-recipe brewing tutor: measured progress

**Not accepted yet.** The known/unknown candidate passes development but fails reserved wording. The larger balanced continuation finishes at 3/5 original development questions and 0/2 beginner explanations. A focused short-answer teaching run is underway. Scaling remains paused.

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
| Short-answer teacher | Pending | Now a regression suite | In progress |

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
| Short-answer teacher | 132 / 332 replayed | 332 planned | Pending | Pending | Known/unknown adapter |

The first three runs total 97.3 training minutes. Mean losses are respectively 0.398605, 0.107404 and 0.411845; changed curricula make them unsuitable as a ranking of answer quality. No examples were silently truncated or skipped. All previous datasets, adapters, checkpoints, raw answers and logs remain preserved.

Continuation loads existing LoRA weights with `is_trainable=True`, checked against the installed PEFT implementation and [official API](https://huggingface.co/docs/peft/v0.13.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained). Parent model identity, revision, LoRA settings and weight/config/prompt hashes enter the run signature. Each continuation has a fresh optimizer; interrupted runs resume their own saved optimizer.

## Balanced question coverage and remaining gates

The balanced curriculum has 18 full-guide requests, 12 canonical follow-ups, 76 direct variants, 30 conversational examples and 14 known/unknown contrasts. It covers every stage, short questions without the recipe name and follow-ups after a full guide. All complete-guide targets retain the original source facts. There are 22,741 supervised answer tokens per pass; the longest conversation is 971 tokens. All 150 examples fit.

This run starts from the first complete-guide adapter at learning rate 0.00005, saving each epoch. The first epoch saved at step 150 after approximately 44.4 minutes and was deliberately stopped before epoch-end validation. No final training metrics or validation loss are claimed for that interrupted phase. The generated complete guide passes, but four development follow-ups fail: ingredients and mash receive unrelated water/settings answers, cold stages omit cooling/pitching/primary, and known/unknown settings falsely deny the primary schedule. The planned second epoch resumes the same run, data, optimizer and scheduler from checkpoint 150 using `--resume`. The first checkpoint and its failed answers remain preserved.

The original final set is consumed and now becomes an additional regression gate. Four new confirmation phrasings were frozen before balanced training; the original four-turn actual-history dialogue remains reserved. No evaluation question is copied verbatim into training. Acceptance also includes two direct explanation-development questions under different wording, so the model must demonstrate basic understanding before reserved chat generation. Acceptance requires all five original development questions, both explanation-development questions, all four regression questions, all four fresh confirmation questions and all four original dialogue turns plus two fresh beginner-explanation turns to pass as one model. This tightens coverage without relaxing any factual requirement.

The six-turn natural dialogue retains the original four questions in order, then asks for plain meanings of wort/pitching and the reason for cleaning and sanitizing. It was frozen before checkpoint selection. The dialogue feeds each model its own generated replies. Reference answers are saved for review and never inserted into history. Both confirmation and dialogue concern the one trained recipe, not unseen recipe families. Greedy decoding, model pin, system prompt, loader/evaluator and suite hashes must match between before/after runs.

All 67 software tests pass, including frozen inputs, question separation, source restrictions, continuation signatures and interrupted dialogue resumption without reference leakage. Data and weights remain local and ignored by Git. See [the reproduction commands](../docs/SINGLE-RECIPE.md). A successful bounded demonstration would still need broader testing and expert review before expansion.


## Completed balanced run and focused correction

The balanced run completes step 300. It preserves the complete guide and repairs the ingredient and mash answers, scoring 3/5 development. The post-boil question still receives only kegging/gelatin/lagering. The known/unknown answer invents a four-gallon batch and packaging instructions and treats the provided batch size as absent. Both explanation questions are off-topic: wort/pitching gets a boil answer, and cleaning/sanitizing gets a lagering answer. This model is rejected before any fresh confirmation or natural-dialogue answers are generated.

The first phase reports 44:24 of progress time. The resumed phase records 2,654.6 project-monotonic seconds (44.2 minutes) but 60,523.3 Trainer elapsed seconds (16.8 hours), with a long overnight progress gap. Do not describe the latter as continuous compute. Resume-phase mean loss 0.066783 is not a whole uninterrupted-run comparison. Recorded peak RSS during resume is 16.02 GiB. All 150 rows fit.

The next run returns to the known/unknown adapter, which already passed the original five development questions, rather than inheriting the rejected balanced weights. Its 132 examples include ten full guides, twelve canonical follow-ups, 76 recipe follow-up variants, twelve plain-language concept variants, fourteen known/unknown contrasts and eight conversations. One hundred short examples are replayed three times: 332 optimizer steps, not 332 distinct recipes or questions. Cold-stage and completion answers start with the requested action; known settings and real gaps are separated clearly. General explanations define wort/pitching and distinguish cleaning from sanitizing, including two follow-ups after a complete guide.

This is standard assistant-only LoRA training with the existing replay sampler, learning rate 0.0001 and one pass over the weighted dataset. No custom loss, factual system prompt or lookup is introduced. All full-guide source facts remain intact; evaluation files are copied unchanged. All 132 examples / 332 replayed rows fit, with 32,047 supervised answer tokens and maximum 880 tokens. The original five development questions, two explanation-development questions and consumed four-case regression set must pass before selecting the model. The four confirmation questions and six-turn actual-history conversation are still unopened.
