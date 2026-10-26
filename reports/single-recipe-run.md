# One-recipe brewing tutor: measured progress

**Not accepted yet.** A Helicious Helles adapter passes all five development questions but only one of four reserved phrasings. A balanced continuation is running. No scaling or general brewing-accuracy claim follows from these results.

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
| Balanced continuation | Pending | Now a regression suite | In progress |

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
| Balanced questions | 150 | Up to 300 planned | Pending | Pending | Complete-guide adapter |

The first three runs total 97.3 training minutes. Mean losses are respectively 0.398605, 0.107404 and 0.411845; changed curricula make them unsuitable as a ranking of answer quality. No examples were silently truncated or skipped. All previous datasets, adapters, checkpoints, raw answers and logs remain preserved.

Continuation loads existing LoRA weights with `is_trainable=True`, checked against the installed PEFT implementation and [official API](https://huggingface.co/docs/peft/v0.13.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained). Parent model identity, revision, LoRA settings and weight/config/prompt hashes enter the run signature. Each continuation has a fresh optimizer; interrupted runs resume their own saved optimizer.

## Balanced question coverage and remaining gates

The balanced curriculum has 18 full-guide requests, 12 canonical follow-ups, 76 direct variants, 30 conversational examples and 14 known/unknown contrasts. It covers every stage, short questions without the recipe name and follow-ups after a full guide. All complete-guide targets retain the original source facts. There are 22,741 supervised answer tokens per pass; the longest conversation is 971 tokens. All 150 examples fit.

This run starts from the first complete-guide adapter at learning rate 0.00005, saving each epoch. The first fully saved epoch will be inspected before spending a second epoch. It can resume from that checkpoint with the original command plus `--resume`.

The original final set is consumed and now becomes an additional regression gate. Four new confirmation phrasings were frozen before balanced training; the original four-turn actual-history dialogue remains reserved. No evaluation question is copied verbatim into training. Acceptance also includes two direct explanation-development questions under different wording, so the model must demonstrate basic understanding before reserved chat generation. Acceptance requires all five original development questions, both explanation-development questions, all four regression questions, all four fresh confirmation questions and all four original dialogue turns plus two fresh beginner-explanation turns to pass as one model. This tightens coverage without relaxing any factual requirement.

The six-turn natural dialogue retains the original four questions in order, then asks for plain meanings of wort/pitching and the reason for cleaning and sanitizing. It was frozen before checkpoint selection. The dialogue feeds each model its own generated replies. Reference answers are saved for review and never inserted into history. Both confirmation and dialogue concern the one trained recipe, not unseen recipe families. Greedy decoding, model pin, system prompt, loader/evaluator and suite hashes must match between before/after runs.

All 66 software tests pass, including frozen inputs, question separation, source restrictions, continuation signatures and interrupted dialogue resumption without reference leakage. Data and weights remain local and ignored by Git. See [the reproduction commands](../docs/SINGLE-RECIPE.md). A successful bounded demonstration would still need broader testing and expert review before expansion.
