# BeerLLM

Collect beer recipes and brewing instructions for a future LoRA/QLoRA fine-tuning dataset.

## Fetch recipes

Requires Python 3 and internet access. No packages, account, or API key needed.

```bash
python3 scripts/fetch_recipes.py
```

This fetches the entire paginated [PunkAPI v3](https://github.com/alxiw/punkapi) catalog. For a small sample:

```bash
python3 scripts/fetch_recipes.py --limit 10
```

Inspect the results:

```bash
python3 -m json.tool data/punkapi/report.json
head -n 1 data/punkapi/recipes.jsonl | python3 -m json.tool
```

## What it saves

- `data/punkapi/raw.json`: original recipe objects, including ingredients, quantities, mash steps, fermentation temperature, brewer tips, and special instructions where provided.
- `data/punkapi/recipes.jsonl`: one recipe per line with source URL, fetch timestamp, and completeness audit.
- `data/punkapi/report.json`: field coverage and the number requiring instruction review.

All original fields and units are preserved. The script never generates missing instructions or assumes labels such as “start” and “end” are exact hop timings. Each run replaces these three files after successful downloading. A sample run fetches full API pages and keeps only the requested number of records. Data remains local and is excluded from Git.

## What happens next

Review source instructions for boil duration, cooling/pitching, fermentation completion, conditioning, and packaging. These may appear in free-text tips, but the script deliberately requires review rather than declaring them complete automatically. Check quantities and unit validity as well; the audit currently checks field presence, not brewing correctness.

Only sufficiently complete, reviewed recipes should become full-procedure training examples. This script does not yet generate training examples or train a model. The old GitHub/BeerXML fetcher and its yeast-only QA generation have been removed. Old local `data/raw` and `data/curated` files, if present, are unused.

The repository MIT license covers project code; source recipe reuse terms need separate verification.

## Development

Use a feature branch, open a pull request, then squash merge into `main`. Keep commit messages to one concise line.
