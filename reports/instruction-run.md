# Instructional LoRA experiment

Training and paired evaluation completed on October 19, 2026 (America/Los_Angeles), locally on the Intel MacBook Pro. **The adapter is runnable, but this experiment failed the brewing-teacher goal.** It still invents recipe facts and loops. Lower reference loss did not translate into useful recipe recall or reliable procedures. The adapter is saved at `outputs/instruction-lora-v1/adapter/` for local testing.

## What was trained

The model is the same pinned SmolLM2-360M-Instruct base as the first experiment. It starts from the original base, not the earlier PunkAPI adapter. LoRA updates 819,200 query/value parameters, approximately 0.226% of the model. The earlier adapter and its checkpoints are preserved.

This is an intentionally small learning experiment: **eight source-checked recipes and 34 conversations**, not training on the entire collection. The recipe families are Alt Ale Altbier, Griffin Spit IPA, Bell's Two Hearted Ale Clone, Angie's Bursting Citra II, Düsseldorf Altbier, Helicious Helles, Höfter's Blanc Pale Ale and Black is Back.

The dataset contains eight named-recipe guides, eight supplied-recipe guides, 16 follow-ups and two general process checks. Handwritten plans preserve recipe-specific quantities and timings, explain terms and explicitly identify missing parameters. General sanitation and fermentation-completion guidance is identified separately and linked to its reference. These plans are assistant-authored and source-checked, not expert brewer-reviewed.

Four separate recipe families supply validation references and three supply the final held-out tests. Those references retain source-text directions and have not undergone the same teaching rewrite or expert review. Family separation passed; no final-test recipe is used for training. Pretraining contamination cannot be ruled out.

## Training measurements

| Measurement | Actual result |
|---|---:|
| Training conversations | 34 |
| Overlength examples skipped | 0 |
| Longest training conversation | 1,419 tokens |
| Context limit | 2,048 tokens |
| Configured epochs | 5 |
| Completed optimizer steps | 40 |
| Trainer-reported effective epochs | 4.7059 |
| Wall time | 1,114.4 seconds / 18.6 minutes |
| Mean training loss over the run | 2.0520 |
| Final validation loss | 0.6050 |

The trainer's accumulation/update accounting stops this configuration at 40 steps, with the reported effective epoch below five. Validation losses were 0.6285, 0.6184, 0.6109, 0.6060 and 0.6050. They are per-example trainer averages on supplied-recipe references; they are not directly comparable with the token-weighted benchmark losses below. Training mixes harder closed-book tasks with supplied-context tasks, so a lower validation loss than mean training loss does not itself establish generalization.

## Before/after protocol

All nine baseline answers were generated and saved before training. Model revision, recipe-family partitions, teaching data, training settings, prompts, reference answers, per-case response limits and greedy decoding were fixed. The same nine questions are then evaluated with the adapter. No training setting was changed after reading the baseline answers.

- Four questions ask for recipes or details seen in fine-tuning, with no recipe facts supplied. This tests **seen-recipe recall**, not unseen-recipe knowledge.
- Three questions supply facts from recipe families held out of fine-tuning. This tests **supplied-recipe interpretation**, not closed-book recall.
- One asks for a private recipe whose details were never supplied.
- One paraphrases the general fermentation-completion concept taught during training; it is not an unseen-topic test.

Loss uses teacher forcing and supervises only assistant reference tokens. Category loss is weighted by token count; conditional answer perplexity is its exponential. Low loss can reflect copied source wording or shared guidance rather than brewing competence. The references share some general guidance with training, and several source procedures remain incomplete.

## Results

| Task / metric | Base before | Adapter after |
|---|---:|---:|
| Seen-recipe recall: answer-token cross-entropy | 2.2066 | 2.1138 |
| Seen-recipe recall: conditional answer perplexity | 9.0847 | 8.2796 |
| Seen-recipe recall: pattern checks | 1/16 | 1/16 |
| Held-out supplied recipes: answer-token cross-entropy | 0.9190 | 0.8732 |
| Held-out supplied recipes: conditional answer perplexity | 2.5067 | 2.3946 |
| Held-out supplied recipes: pattern checks | 10/12 | 4/12 |
| Private-recipe uncertainty: conditional answer perplexity | 11.2929 | 10.0809 |
| Private-recipe uncertainty: pattern checks | 1/2 | 0/2 |
| General completion check: conditional answer perplexity | 35.8809 | 30.7114 |
| General completion check: pattern checks | 0/3 | 0/3 |
| All responses reaching their generation limit | 7/9 | 6/9 |

All four task categories have slightly lower reference perplexity. Nevertheless, seen-recipe checks do not improve, supplied-recipe checks worsen, and actual answers remain incorrect. The smaller number of token-limit hits is not a quality success: a shorter answer can still be wrong.

Generation alone took 446.8 seconds before and 437.6 seconds after. These are operational observations on one laptop, not a controlled speed benchmark. Examples vary greatly in response length and prompt size.

### Read the actual answers

Codex inspected all 18 answers against the source facts and requests. This review is assistant-authored, not blinded, independent or expert brewer review. The reviewer also authored the teaching plans. Scores below describe task-specific fact fulfilment: **0** missing or critically wrong; **1** some required facts preserved but incomplete; **2** the requested factual relationships preserved. They are not population accuracy percentages or proof of source recipe validity.

| Question | Before | After | Observed behavior |
|---|---:|---:|---|
| Alt Ale full recipe, no context | 0 | 0 | Both invent gallon-based ingredients; adapter repeats salt additions rather than a usable procedure |
| Griffin fermentation/dry hops, no context | 0 | 0 | Both invent 10–15 lb dry hops instead of the source's 1 oz; temperatures and contact time are wrong |
| Bell's boil timing, no context | 0 | 0 | Both confuse hops with malt extract; adapter places additions at 100 minutes in a 75-minute boil |
| Angie's dry-hop sequence, no context | 0 | 0 | Both invent doses; adapter gives invented fermentation temperatures and repeats stages |
| Dubious Dubbel, facts supplied | 0 | 0 | Both substitute Saaz for the source's Styrian Goldings addition; adapter loops on section numbering instead of producing instructions |
| Black Is Beautiful stout, facts supplied | 1 | 1 | Both preserve most source facts but omit bottling and missing-parameter handling; little explanation is added |
| Gose acid-malt mash, facts supplied | 2 | 0 | Base gives the requested 2 oz / reserved 14 oz sequence, then exceeds scope; adapter copies ingredients and cuts off before answering |
| Private recipe with no facts supplied | 0 | 0 | Both invent grain quantities rather than admit they lack the recipe |
| Fermentation-completion check | 0 | 0 | Neither gives the taught stable-gravity check; adapter repeats bottle-fill statements |

The exact training follow-up “Walk me through Griffin Spit IPA's fermentation and dry hopping” was also tried through the new default chat CLI after the benchmark. The adapter loaded successfully but invented ingredients and looped. This post-hoc diagnostic is **excluded from the nine-case scores**. It shows the failure is not limited to paraphrased questions; fitting the training facts has not been demonstrated. All 64 saved LoRA B matrices have nonzero learned weights, so the experiment did save updated adapter parameters.

### Scoring safeguards and limits

A reference preflight exposed a numeric-pattern bug: a bounded sentence check stopped at the decimal in “3.5 gallons.” The corrected scorer permits decimal points within a quantity while retaining sentence boundaries. Both saved answer sets were rescored under the same rule. All 33 frozen reference checks pass the corrected scorer. The raw generation reports, their signatures and answers remain unchanged; `comparison.json` records the original generation-evaluator hash, current scorer hash and corrected per-case matches.

Even with that fix, these are **pattern checks, not factual accuracy**. For example, the base private-recipe answer matches the word “provide,” yet actually fabricates the recipe rather than asking the user to provide it. Dubbel's correct-looking ingredients and temperatures can also appear alongside a corrupt procedure. Manual inspection is necessary.

## What this establishes

The data → LoRA training → saved adapter → identical before/after evaluation → chat workflow is operational. **This particular model/data/training setup did not learn the intended useful brewing behavior.** It cannot be presented as a reliable beer assistant, a successful factual fine-tune or evidence that the whole collection has been learned.

The result does not prove that local fine-tuning is impossible, that LoRA cannot work or that a larger model alone will fix it. Before another full run, use validation and training-prompt fitting diagnostics, improve the starting model and/or training setup, and expand reviewed teaching data. Any iteration informed by this final test needs a new untouched final test set. Do not keep reporting this same nine-question set as an unseen benchmark.

## Software checks and saved artifacts

All **43 unit tests passed**, along with Python compilation, runner shell syntax and diff checks. Tests cover source/family separation, assistant-only masking, preserved quantities, chat history limits, token-weighted perplexity, identical comparison inputs, decimal boundaries and rescoring without overwriting raw answers. The real one-shot chat command loaded the default adapter and returned an answer; that software check passed while the answer-quality check failed.

Local artifacts:

- `data/instruction_v1/manifest.json`: dataset source/lesson/builder hashes and actual task counts.
- `data/instruction_v1/benchmark.jsonl`: frozen prompts, references, checks and response limits.
- `outputs/instruction-lora-v1/run_config.json`: model revision, training configuration and train/validation hashes.
- `outputs/instruction-lora-v1/train_metrics.json` and `trainer_state.json`: runtime, loss and validation history.
- `outputs/instruction-lora-v1/checkpoint-35/` and `checkpoint-40/`: retained resumable checkpoints.
- `outputs/instruction-v1/{before,after,comparison,answer_review}.json`: complete outputs, metrics and assistant review.
- `outputs/instruction-v1/chat-smoke.txt`: the additional exact-training-prompt CLI diagnostic.

Frozen suite SHA-256: `46453cb1ac1f4576e5b0d3a32fe80630f5f7f3f59df2b79bfce261e1860949a1`.

Adapter weights SHA-256: `8c32607e5f387303f342a07a7c67fd220a4bfe4f61356c28c00a14b85468356e`.

Adapter configuration SHA-256: `7805df31116b65015a5390d27d09ac413bd3a0607d3e9b278e11bd077fa761a4`.

## Reproduce and test

See [the exact experiment commands](../docs/EXPERIMENT.md) and [the metric guide](../docs/METRICS.md). The raw generated answers and references remain local in `outputs/instruction-v1/before.json` and `outputs/instruction-v1/after.json`. Data, model weights and complete source records are ignored by Git.

```bash
# Interactive chat with the new adapter; follow-up history stays in memory.
.venv/bin/python -m beerllm.chat

# One question, with an explicit adapter path.
.venv/bin/python -m beerllm.chat --adapter outputs/instruction-lora-v1/adapter \
  --prompt "Explain Griffin Spit IPA's fermentation and dry-hop stages."

# Compare the original base model on the same question.
.venv/bin/python -m beerllm.chat --base \
  --prompt "Explain Griffin Spit IPA's fermentation and dry-hop stages."
```

This small convenience sample, unreviewed source references and an assistant's own answer review cannot establish brewing reliability. Numeric pattern checks are not factual accuracy. Source procedure gaps and source reuse rights remain unresolved. Do not treat the generated answers as expert-verified brew sheets.
