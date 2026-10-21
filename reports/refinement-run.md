# Refined Qwen LoRA experiment

Training completed on this Intel Mac. The selected Qwen2.5-1.5B adapter passes **5 of 8 development cases, versus 1 of 8 before training**, on assistant inspection of the frozen factual rubric. All four reworded short-fact cases improve from fail to pass. **Both complete recipe guides still fail**, as does the general fermentation-completion check. This is a useful partial learning result, not a reliable brewing teacher.

## Matched development comparison

The base and adapter use the same pinned model revision, exact system prompt, questions, source notes, references, output limits and neutral greedy decoding. The comparator verifies suite, evaluator and loader hashes. This refined-prompt baseline is separate from the [earlier negative Qwen trial](instruction-v2-run.md); its older prompt/decoder results are not relabeled as matched controls.

| Task | Before loss | After loss | Before perplexity | After perplexity |
|---|---:|---:|---:|---:|
| Four seen facts, new wording | 2.276 | 0.328 | 9.742 | 1.388 |
| Complete seen-recipe guide | 2.056 | 0.467 | 7.816 | 1.595 |
| Supplied validation recipe | 0.755 | 0.184 | 2.128 | 1.202 |
| Unavailable private recipe | 2.705 | 1.142 | 14.957 | 3.132 |
| Completion check | 4.192 | 2.241 | 66.175 | 9.402 |

Loss is assistant-token cross-entropy under teacher forcing; perplexity is its exponential. Each category is weighted by reference answer tokens. Lower values measure better prediction of reference wording, not correct autonomous generation. The guide's perplexity falls from 7.816 to 1.595 even though its actual answer invents ingredients and loops. The [metric guide](../docs/METRICS.md) explains this distinction.

| Strict full-answer review | Before | After |
|---|---:|---:|
| Reworded facts: volumes, lager stages, mash stages, dry hopping | 0/4 | 4/4 |
| Complete named-recipe guide | 0/1 | 0/1 |
| Complete guide with validation recipe supplied | 0/1 | 0/1 |
| Unavailable private recipe | 1/1 | 1/1 |
| General completion check | 0/1 | 0/1 |
| Total | 1/8 | 5/8 |

This is a small convenience sample reviewed by the assistant, not independent expert review or population brewing accuracy. The Qwen suites have no lexical pattern checks; their pattern-check rate is null, not a zero-percent accuracy score. Token-limit hits fall from two of eight to one of eight, but that does not make incomplete answers correct.

Concrete findings:

- The Helles answer correctly separates primary fermentation at 50°F / 10°C for two to three weeks from later lagering at 40°F / 4°C for three to six weeks. The volume answer preserves 3 gallons / 11.4 L in the kettle and 5.5 gallons / 21 L after dilution.
- The SMaSH guide invents a multi-extract recipe, then repeats fabricated ingredient/yeast-number lines until the 900-token limit. Its repeated-four-gram fraction is 0.676. It never produces a complete procedure.
- The supplied Ardley answer copies the ingredient list, but its procedure omits hop additions, invents a dry-hop dose, misplaces yeast, and falsely says the source lacks explicit fermentation/carbonation/priming details. Correct ingredient copying does not pass a whole guide.
- The private-recipe answer appropriately requests notes. The completion answer instead talks about a Centennial hop dose and fails to request stable repeated gravity readings.

Full answers, signatures and review reasons remain local in `outputs/instruction-v3/development-{before,after,comparison}.json` and `answer-review.json`.

## Data, training and checkpoint selection

The frozen refined release contains 197 distinct conversations from eleven source-checked training recipes, with four validation families and two reserved final families. These are assistant-authored/source-checked references, **not expert brewer-reviewed brew sheets**. Source procedure gaps and typos remain explicit. The model does not train on all 1,498 collected recipes.

Tasks include ingredient lists, ordered guides, direct factual questions, alternate question/answer wording, source notes, conversational follow-ups and uncertainty/completion examples. The 88 short factual rows appear twice per epoch, so training samples 285 rows per epoch. Replays are sampling weights, not additional independent examples. All sampled rows fit the 2,048-token budget; none are truncated or skipped.

The full run starts from pinned Qwen revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, not from a tiny fitting adapter. LoRA rank is 16, alpha 32, dropout zero, with attention and feed-forward targets: 18,464,768 trainable parameters (about 1.18%). Learning rate is 0.001 with the configured schedule. Training uses float32 CPU execution and four threads; this is LoRA, not QLoRA.

All 855 steps / three epochs finished in 9,763.4 seconds (2 hours 43 minutes). Mean training loss is 0.525402; peak process RSS is 19.10 GiB on the 32-GiB machine.

| Epoch | Validation reference loss |
|---|---:|
| 1 | 0.188804 |
| 2 | 0.157951 |
| 3 | 0.233387 |

Before generating full development/final answers, checkpoint 570 (epoch two) was selected by lowest epoch-end validation loss. This rule was recorded after training, not claimed as preregistered. The third epoch's worse validation prediction makes retaining the earlier candidate reasonable; it does not prove that every answer improves at that checkpoint.

The selected files are at `outputs/qwen-instruction-v3/best-adapter/`, with the matching saved system prompt. Weight SHA-256 is `e0d43de93c76b734572c11c97f998c8df4315bce72d179a00c522761fbe94556`. Local `selection.json` records the criterion and checkpoint. The final epoch remains at `outputs/qwen-instruction-v3/adapter/`; all nine optimizer checkpoints and previous experiments are preserved.

## Fitting gate and generated probes

A preceding twelve-example / two-fact fitting run passed two exact questions and two fixed rewordings after 60 steps (10.8 minutes; 15.46 GiB peak RSS). Those are trained facts, not unfamiliar recipes. Multiple changes occurred together, so the result does not isolate one cause. Its tiny adapter is not promoted.

The full run generated three development probes at every checkpoint: exact volume, reworded volume and reworded Helles stages. They cover only two trained facts.

| Step | Passing full answers | Observation |
|---|---:|---|
| 95 | 0/3 | Wrong batches and fermentation parameters |
| 190 | 0/3 | Wrong quantities, units and stages |
| 285 | 0/3 | Retrieves Helles values but places primary after kegging |
| 380 | 0/3 | Cross-recipe interference |
| 475 | 2/3 | Rewordings pass; exact answer adds a confusing sentence |
| 570 | 3/3 | All requested values and relationships correct |
| 665 | 2/3 | Reworded volume regresses into hop instructions |
| 760 | 3/3 | Volume answer recovers |
| 855 | 3/3 | Final probes remain correct |

Improvement is not monotonic. These passing short probes cannot establish complete-guide quality. Raw checkpoint answers and inspection notes are preserved in `outputs/qwen-instruction-v3/probes/` and `outputs/instruction-v3/probe-review.json`.

## Reserved final families

After selection and the full development review, the same fixed model/base pair is evaluated on Anchor Steam Clone and Holier than Meow Hefeweizen. Their families are excluded from this project's teaching data; both models receive identical actual source notes. This tests interpretation, not recall of never-supplied recipe facts. Public base pretraining exposure is unknown.

**Status:** training and matched development comparison complete; reserved final evaluation is running. No adapter is promoted as a reliable brewing assistant.

## Test locally and reproduce

See [the exact training, selection and evaluation commands](../docs/REFINEMENT.md). For a focused local test:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m beerllm.chat \
  --adapter outputs/qwen-instruction-v3/best-adapter \
  --prompt 'List Helicious Helles primary fermentation and later cold storage as two stages, with the temperature and duration of each.'
```

Omit `--prompt` for interactive chat. Add `--recipe-title 'Helicious Helles'` to explicitly supply saved source notes. Add `--base` to compare the selected adapter's starting model and saved system prompt without adapter weights. Source context is not evidence that the weights memorized a recipe. The old no-argument adapter default remains unchanged; select this experiment explicitly.

All 60 unit tests pass, including masking, source/context integrity, held-out-family guards, neutral decoding, varied-fit selection, replay weights, prompt metadata and chat history checks. Compilation and diff checks pass. Generated data, weights, full answers, caches and logs remain local and ignored by Git.

The next quality gate should require a complete guide to fit under both exact and reworded requests before another broad run. Full-guide/source-context exposure and general completion checks need attention alongside short facts. Further experiments should change factors separately and reserve new final families once these final answers have been inspected. More epochs alone are not supported by the observed validation regression.
