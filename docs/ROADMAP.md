# Roadmap: a conversational brewing teacher

## Target behavior

A brewer asks for a recipe and receives the ingredients, quantities, ordered instructions, plain-language explanations, common mistakes and checks for completion. The model should speak naturally and preserve recipe facts. No raw database record should be required in the user's normal question. Missing facts must be acknowledged rather than invented.

## Technology

Python, Requests, Beautiful Soup, local HTML snapshots and JSONL. Use argparse commands and the existing PyTorch/Transformers/PEFT LoRA trainer. Start with selected public AHA recipe URLs. No database, Spark, Airflow, Kafka or cloud service is necessary for this volume. Add orchestration only if recurring multi-source collection warrants it.

## Milestones

| Phase | Work | Acceptance check | Status |
|---|---|---|---|
| 1. Source extraction | Fetch selected public AHA recipe pages, cache HTML and preserve original ingredient/direction text | Every extracted number and instruction can be traced to its source; unavailable pages fail visibly | Implemented; five-page pilot run |
| 2. Completeness review | Review batch size, ingredient quantities, mash/steep, boil and additions, cooling/pitching, fermentation and packaging | Identify exact gaps and contradictions; stage keywords alone never approve a recipe | First 1,498-record audit complete: 50 candidates; source review and enrichment pending |
| 3. Training conversations | Convert recipes into natural user requests and assistant answers; then author supported explanations, follow-ups and mistakes-to-avoid answers | No JSON in normal user prompts; accurate source facts; useful beginner language; no invented instructions | Refined release: 197 distinct conversations from eleven source-checked training recipes; expert review pending |
| 4. Freeze dataset | Deduplicate related recipes, keep recipe families together, freeze source hashes and splits | No family leakage; all approved conversations fit the chosen context window intact | Refined release freezes eleven training/four validation/two reserved final families; all sampled training examples fit 2,048 tokens; not an expert-approved release |
| 5. Small training experiment | Verify tokenization, assistant-only loss and a short LoRA run on the new format | Real data loads and trains; inspect whether a tiny set can produce prose and ordered instructions | Earlier exact-only checks failed rewordings; varied Qwen fitting check passes two exact questions and two rewordings covering two trained facts |
| 6. Full fine-tuning | Choose model size after a local benchmark; train on the reviewed release with checkpoints | Save adapter/config/data hashes; verify generated examples during validation | First Qwen trial failed strict review; refined 1.5B CPU run finished three epochs in 2h43m; epoch two selected by validation loss; all checkpoints and final weights retained |
| 7. Evaluate and use | Compare baseline and adapter for correctness, clarity, procedure completeness and follow-up answers | Human inspection of ingredient-quantity pairs, temperatures, timings and explanations; detect copying and repetition | Refined matched development review: 1/8 before, 5/8 after; complete guides still fail; two reserved-family comparisons running; independent expert review pending |

## Dataset design

One JSONL line is one conversation with a `messages` list. Each message contains `role` and `content`. Source URL, source HTML hash, family ID and review status remain metadata outside the messages. Only the assistant's answer tokens contribute to training loss.

The pilot uses a natural request for a named recipe and an answer containing the source ingredient list and original directions in order. This proves ingestion and model-input compatibility. It does not independently verify the recipe or create new explanations of why each action works.

The next curation pass should create source-supported conversations for:

- A complete recipe request.
- A beginner asking what a particular instruction means.
- A follow-up about why the step matters.
- A common mistake, its likely consequence and a supported corrective action.
- An underspecified question that needs clarification.

Do not multiply near-identical templates to inflate dataset size. Keep every conversation derived from the same recipe family in the same partition. The pilot split allocation can change as recipes are added; freeze it before an actual experiment.

Evaluate named-recipe recall on recipes seen in training using new questions, and report it explicitly as recall. For unseen recipes, supply facts when testing interpretation or test whether the model admits it lacks the recipe; do not claim factual recall of never-supplied records. Exact recipe reproduction and general brewing competence are separate capabilities.

## Run the pilot

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m beerllm.aha --urls-file configs/aha-sample.txt
.venv/bin/python -m beerllm.aha_dataset
.venv/bin/python -m beerllm.train --data-dir data/aha/training_candidates --check-data
```

Outputs are under `data/aha/`. Raw HTML is cached so reruns reuse it. `recipes.jsonl` preserves source text, `report.json` records imports and errors, and `training_candidates/` contains the model-format examples and a split report. These files stay local and are excluded from Git. Each import replaces that batch's derived recipe/report files while retaining cached pages.

The importer observes robots.txt, waits at least ten seconds between network requests, and stops short of following login/redirect flows. It needs no membership credentials and does not perform a bulk crawl. Technical accessibility does not establish a training license; source reuse terms still need confirmation before scaling or redistribution.

Do not replace the old experiment or start a long training job until the new candidate examples have passed review. Use a separate output directory for every experiment.

The multi-source collection is now capped and stopped. The [curation guide](CURATION.md) and [audit report](../reports/curation-v1.md) describe the next-stage commands, exact counts and remaining teaching-data work.

The [instructional run report](../reports/instruction-run.md) records actual training and before/after results. The [experiment guide](EXPERIMENT.md) has reproducible commands and the [metric guide](METRICS.md) explains what the scores can and cannot establish.

The [training-fit diagnostic](../reports/training-fit-check.md) demonstrates why exact memorization is insufficient. The [first Qwen report](../reports/instruction-v2-run.md) records a negative broader trial and the subsequent varied-fitting gate. The [refined run report](../reports/refinement-run.md) tracks the current broader curriculum, matched baseline and factual checkpoint reviews. See [the refined experiment guide](REFINEMENT.md) for commands. Small fitting checkpoints are not promoted as brewing assistants.
