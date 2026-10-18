# Collect public beer recipes

The collector discovers public recipe URLs from AHA and Brewer's Friend sitemaps, imports the local BrewDog/PunkAPI catalog, downloads accessible recipe pages, and exports a combined collection. It does not train a model or invent missing instructions.

## Commands

```bash
# Discover current public recipe URLs and import BrewDog records.
.venv/bin/python -m beerllm.collection discover

# Test a bounded batch: up to 30 pending pages per website.
.venv/bin/python -m beerllm.collection run --limit 30

# Process the entire remaining queue, resuming saved progress.
# On macOS, keep the machine awake while the command runs:
caffeinate -i .venv/bin/python -u -m beerllm.collection run --limit 0 > outputs-collection.log 2>&1
```

Check progress from another terminal:

```bash
.venv/bin/python -m beerllm.collection status
tail -f outputs-collection.log
```

Request a graceful stop (the current request may finish):

```bash
touch data/collection/STOP
```

To resume, remove only that project-local stop flag and run the collection command again. Finished records are skipped. Parse errors are preserved separately rather than retried endlessly; rate-limited requests use a saved cooldown. Only one collecting process may run at a time; a file lock prevents duplicate workers. Discovery can be rerun to add new URLs.

## What is saved

- `data/collection/queue.sqlite3`: a standard-library SQLite file for URLs, per-page status, errors and parsed records. This is local bookkeeping, not a database server or a new service.
- `data/collection/raw/`: gzip-compressed original HTML, so completed downloads can be reparsed without another web request.
- `data/collection/sitemaps/`: compressed source discovery snapshots.
- `data/collection/unique_recipes.jsonl`: combined exact-content-unique records, with all matching source links retained.
- `data/collection/similar_recipe_groups.jsonl`: possible related recipes requiring review. They are retained in the main collection.
- `data/collection/report.json`: counts, queue status, exact duplicates and instruction coverage at the last export.

The process exports snapshots periodically and when a bounded run ends. For a fresh snapshot on demand:

```bash
.venv/bin/python -m beerllm.collection export
```

Raw data, SQLite files and exports are ignored by Git. The existing pilot datasets and adapters are preserved. Collection stops if available project disk space falls below 5 GiB.

## Sources and limits

Discovery on October 18, 2026 found 327,867 Brewer's Friend public recipe URLs and 1,528 AHA public recipe URLs. These are discovery counts, not successfully downloaded or complete recipes. The BrewDog API snapshot contains 415 records.

AHA requests are spaced at least ten seconds apart; Brewer's Friend requests at least five seconds apart. One worker per website runs sequentially, with independent websites collected in parallel. The complete initial queue therefore takes roughly 4.3 hours for AHA and at least 19 days for Brewer's Friend, before response time, downtime or failures. Keep the laptop powered and awake for uninterrupted collection; restarting resumes successful work.

The collector follows robots.txt, avoids disallowed XML/JSON export routes, does not follow login/redirect flows, and pauses a source on 401/403/429. HTTP 429 triggers a persisted cooldown of at least one hour (or longer if Retry-After requires it), then resumes; three consecutive access failures require review. HTTP 401/403 require review. These pauses survive restarts. Some public URLs have unavailable or restricted content. Missing instructions and parse failures stay visible in the queue; there is no paywall bypass. Source reuse terms are not established by public accessibility.

Brewer's Friend supplies Recipe JSON-LD and a public text representation containing additional details such as hop quantities/timings, yeast, mash information and notes. We preserve both. Those structured directions must not be confused with independently authored explanations or a fully complete brew sheet. BrewDog records preserve structured methods and brewer tips even where full written procedures are absent.

## What “unique” means

Exact deduplication compares source quantities, yield, directions, process text and notes after case/whitespace normalization. Titles and author names alone do not establish identity. Differences in recorded ingredients or procedures are preserved. Exact duplicates retain all provenance links.

Different unit notation, paraphrased instructions or differently represented cross-source recipes may remain separate. Similar normalized titles are review groups, not grounds for automatic deletion. This deliberately avoids erasing legitimate variants. More aggressive ingredient/volume normalization and reviewed near-duplicate matching belong to the curation stage before dataset splits.

## After collection

Review completeness and language, normalize quantities with traceable conversions, assess near duplicates, and approve supported instructional conversations. A large recipe collection does not automatically become a complete, high-quality fine-tuning dataset.
