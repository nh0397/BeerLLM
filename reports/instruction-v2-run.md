# Instructional run v2

Started October 21, 2026 (America/Los_Angeles). This report will record the completed local run; the base evaluation is complete, while training and after-evaluation are still pending.

## Frozen plan

Use Qwen2.5-1.5B-Instruct at commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, CPU float32, four threads. Its 1,543,714,304 parameters load locally. The initial three-question capability probe reached about 7.0 GiB peak process RSS. It still invented a private recipe, broadened a source's fermentation target and gave unreliable completion advice. A larger starting model alone is not an adequate fix.

The next dataset has 138 conversations across eleven AHA training recipes, with four separate validation families. The original eight authored recipes are expanded with distinct factual-question wordings, prior-conversation context, ingredient requests and explicit gaps. Three new recipes add examples distinguishing strike/mash temperatures, dry-hop stages and carbonation from sugar. These examples reuse source facts; 138 conversations are not 138 independent recipes.

The original cached HTML hashes and re-extracted ingredients/directions match for all seventeen selected training, validation and final sources. The lessons are assistant-checked against those records, not expert brewer-reviewed. Source typos and procedure gaps remain explicitly identified; automatic verification checks extraction integrity, not brewing accuracy.

The plan uses two epochs, batch size one, no accumulation, learning rate 0.0003, LoRA rank 16/alpha 32/dropout 0.05 on query, key, value and attention output projections. These choices are exploratory, not a proven optimum. Save every 46 optimizer steps and retain six checkpoints plus generated probes. The model must be reviewed on actual answers, not promoted solely for lower reference loss.

Eight frozen development cases include new wordings of seen facts, a complete seen-recipe guide, one supplied validation recipe, uncertainty and a completion check. The new final evaluation reserves Anchor Steam Clone and Holier than Meow Hefeweizen from training and development. It supplies their original recipe notes and measures interpretation, not closed-book recall of unseen facts. Final answers will be generated only after development choices are finished, and will not be used for further tuning in this experiment.

Before/after results compare the same pinned model, prompts, references and greedy decoding. Assistant-token cross-entropy and conditional perplexity are secondary measurements. Separate answer inspection records quantity/stage relationships, omissions, unsupported claims, scope and clarity. Empty lexical-check lists deliberately avoid presenting keyword matches as factual accuracy.

## Baseline and early checkpoints

All eight base development answers fail the strict [factual review criteria](../configs/instruction-v2-rubric.json) on assistant inspection. This is a small convenience sample, not a population accuracy estimate or independent brewer review. The base invents the named recipes and the unavailable private recipe. Even with Ardley Brown Ale supplied, it omits hop doses, invents water amounts and turns the source's batch priming-sugar dose into a per-gallon dose. A lexical match on the sugar number would miss that error.

| Development task | Base reference perplexity |
|---|---:|
| New wordings of seen recipe facts (four cases) | 9.428 |
| Complete seen-recipe guide | 7.990 |
| Supplied validation recipe | 2.184 |
| Unavailable private recipe | 18.792 |
| Completion check | 75.526 |

Six of eight answers reach their token limits. Base generation takes about 16 minutes for the eight cases, including reference-loss calculations and loading; peak process RSS is 9.14 GiB. Raw results are in `outputs/instruction-v2/development-before.json`.

At checkpoints 46, 92 and 138, the exact-volume question and both new-wording probes still fail their requested factual relationships. Responses have become shorter, but incorrect quantities and temperatures remain. Those checkpoints are not promoted. Full probe answers and assistant inspection notes are preserved locally. Epoch-one validation loss is 0.155856 across four supplied-recipe examples; this is teacher-forced loss, not a generated-answer correctness score.

## Reproduction and artifacts

See [the experiment commands](../docs/INSTRUCTION-V2.md). Local data, downloaded models, adapters, full generated answers and logs remain ignored by Git. The previous adapters, source collection and datasets are preserved. The old default chat adapter is unchanged pending review.
