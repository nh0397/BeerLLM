# Instructional run v2

Started October 21, 2026 (America/Los_Angeles). This report will record the completed local run; the 138-conversation trial and its before/after evaluation are complete. It failed the factual quality gate. A revised curriculum is undergoing a small fitting check before another full experiment.

## Frozen plan

Use Qwen2.5-1.5B-Instruct at commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, CPU float32, four threads. Its 1,543,714,304 parameters load locally. The initial three-question capability probe reached about 7.0 GiB peak process RSS. It still invented a private recipe, broadened a source's fermentation target and gave unreliable completion advice. A larger starting model alone is not an adequate fix.

The next dataset has 138 conversations across eleven AHA training recipes, with four separate validation families. The original eight authored recipes are expanded with distinct factual-question wordings, prior-conversation context, ingredient requests and explicit gaps. Three new recipes add examples distinguishing strike/mash temperatures, dry-hop stages and carbonation from sugar. These examples reuse source facts; 138 conversations are not 138 independent recipes.

The original cached HTML hashes and re-extracted ingredients/directions match for all seventeen selected training, validation and final sources. The lessons are assistant-checked against those records, not expert brewer-reviewed. Source typos and procedure gaps remain explicitly identified; automatic verification checks extraction integrity, not brewing accuracy.

The plan uses two epochs, batch size one, no accumulation, learning rate 0.0003, LoRA rank 16/alpha 32/dropout 0.05 on query, key, value and attention output projections. These choices are exploratory, not a proven optimum. Save every 46 optimizer steps and retain six checkpoints plus generated probes. The model must be reviewed on actual answers, not promoted solely for lower reference loss.

Eight frozen development cases include new wordings of seen facts, a complete seen-recipe guide, one supplied validation recipe, uncertainty and a completion check. The new final evaluation reserves Anchor Steam Clone and Holier than Meow Hefeweizen from training and development. It supplies their original recipe notes and measures interpretation, not closed-book recall of unseen facts. Final answers will be generated only after development choices are finished, and will not be used for further tuning in this experiment.

Before/after results compare the same pinned model, prompts, references and greedy decoding. Assistant-token cross-entropy and conditional perplexity are secondary measurements. Separate answer inspection records quantity/stage relationships, omissions, unsupported claims, scope and clarity. Empty lexical-check lists deliberately avoid presenting keyword matches as factual accuracy.

## Baseline and early checkpoints

All eight base development answers fail the strict [factual review criteria](../configs/instruction-v2-rubric.json) on assistant inspection. This is a small convenience sample, not a population accuracy estimate or independent brewer review. The base invents the named recipes and the unavailable private recipe. Even with Ardley Brown Ale supplied, it omits hop doses, invents water amounts and turns the source's batch priming-sugar dose into a per-gallon dose. A lexical match on the sugar number would miss that error.

| Development task | Base reference perplexity | Adapted reference perplexity |
|---|---:|---:|
| New wordings of seen recipe facts (four cases) | 9.428 | 1.494 |
| Complete seen-recipe guide | 7.990 | 3.883 |
| Supplied validation recipe | 2.184 | 1.160 |
| Unavailable private recipe | 18.792 | 10.692 |
| Completion check | 75.526 | 14.091 |

Six of eight base answers reach their token limits; none of the adapted answers do. Both versions still fail all eight strict case reviews. The adapted model copies Ardley ingredients but corrupts its brewing procedure and packaging doses. Its much lower reference perplexity does not establish reliable generation. Base generation takes about 16 minutes for the eight cases, including reference-loss calculations and loading; peak process RSS is 9.14 GiB. Raw results are in `outputs/instruction-v2/development-before.json`.

At all six checkpoints (46, 92, 138, 184, 230 and 276), the exact-volume question and both new-wording probes still fail their requested factual relationships. Responses have become shorter, but incorrect quantities and temperatures remain. Those checkpoints are not promoted. Full probe answers and assistant inspection notes are preserved locally. Validation loss is 0.155856 at epoch one and 0.097959 at epoch two across four supplied-recipe examples. These are teacher-forced losses, not generated-answer correctness scores. The completed 276-step run took 4,068.1 seconds (67.8 minutes), with mean training loss 1.235754 and peak process RSS 18.24 GiB. No training examples were skipped.

## Decoder diagnostic

Qwen's pinned generation preset includes `repetition_penalty=1.1`. The first matched before/after pair inherited that same value. [Transformers documents that this penalty includes prompt tokens](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/generation/logits_process.py#L308), which is undesirable when exact recipe facts need copying. A controlled diagnostic disables the penalty on the same trained adapter: all four reworded recall cases still fail. In a supplied Helicious extraction, disabling it recovers the two correct temperatures but omits durations. This setting was not the sole cause of failure.

Future generation explicitly records neutral greedy settings, including repetition penalty 1.0. Old paired outputs are preserved. A new paired baseline is required for a revised prompt/decoder; comparisons will not attribute a decoding change to training.

## Next fitting gate

The broad attention-adapter run has not learned the checked factual answers. Before choosing another full training run, use `configs/qwen-fit-check.json` on two existing training follow-ups. This diagnostic includes feed-forward projections (`gate_proj`, `up_proj`, `down_proj`) as well as attention, a higher learning rate, no dropout and many passes over two examples. These changes occur together, so success would not isolate one cause. Its repeated-training validation is deliberately not a generalization score, and the tiny adapter will not replace the chat default.

The wider two-answer fitting run completed in 469.5 seconds (7.8 minutes), with 18,464,768 trainable parameters and 8.87 GiB peak process RSS. It answers both exact training questions correctly but fails both fixed rewordings. Its post-training reference loss on the two exact prompts is about 0.000029. This is memorization, not a generalization result, and the tiny adapter is not promoted. Multiple training settings changed together, so this does not isolate the effect of feed-forward adaptation.

A frozen refinement now has 197 conversations from the same eleven recipes. It uses direct answer variants, source-specific factual Q&A, clearer scope instructions and additional uncertainty/completion examples. The new system prompt is stored with future adapters so chat can use the training prompt. A twelve-example variant of the two-fact fitting check is running before the next full configuration is selected. All 58 tests pass.

The reserved final recipe families have not been evaluated. They remain available for the selected full experiment after development choices finish.

## Reproduction and artifacts

See [the experiment commands](../docs/INSTRUCTION-V2.md). Local data, downloaded models, adapters, full generated answers and logs remain ignored by Git. The previous adapters, source collection and datasets are preserved. The old default chat adapter is unchanged pending review.
