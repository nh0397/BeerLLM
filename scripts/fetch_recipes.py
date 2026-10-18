"""Download PunkAPI recipes and audit brewing details. No dependencies required."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

API = 'https://punkapi.dev/v3/beers'
OUTPUT = Path(__file__).resolve().parents[1] / 'data' / 'punkapi'


def quantity(value):
    return (isinstance(value, dict)
            and isinstance(value.get('value'), (int, float))
            and not isinstance(value.get('value'), bool)
            and bool(value.get('unit')))


def audit(recipe):
    ingredients = recipe.get('ingredients') or {}
    method = recipe.get('method') or {}
    mash = method.get('mash_temp') or []
    checks = {
        'batch_volume': quantity(recipe.get('volume')),
        'malt_amounts': bool(ingredients.get('malt')) and all(
            x.get('name') and quantity(x.get('amount')) for x in ingredients.get('malt', [])),
        'hop_amounts': bool(ingredients.get('hops')) and all(
            x.get('name') and quantity(x.get('amount')) for x in ingredients.get('hops', [])),
        'hop_addition_labels': bool(ingredients.get('hops')) and all(
            x.get('add') is not None and str(x['add']).strip() != ''
            for x in ingredients.get('hops', [])),
        'yeast': bool(ingredients.get('yeast')),
        'mash_schedule': bool(mash) and all(
            quantity(x.get('temp')) and isinstance(x.get('duration'), (int, float))
            and not isinstance(x.get('duration'), bool) and x['duration'] > 0 for x in mash),
        'fermentation_temperature': quantity((method.get('fermentation') or {}).get('temp')),
        'brewer_tips': bool(recipe.get('brewers_tips')),
    }
    return {
        'available_fields': [key for key, available in checks.items() if available],
        'missing_fields': [key for key, available in checks.items() if not available],
        # These details may occur in prose. Do not infer their presence from keywords.
        'needs_instruction_review': ['boil_duration', 'exact_hop_timing', 'cooling_and_pitching',
                                     'fermentation_completion', 'conditioning', 'packaging'],
        'status': 'needs_review',
    }


def download(limit=None):
    recipes, seen = [], set()
    page = 1
    while limit is None or len(recipes) < limit:
        url = f'{API}?page={page}&per_page=80'
        request = urllib.request.Request(url, headers={'User-Agent': 'BeerLLM/0.1'})
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.load(response)
        if not isinstance(batch, list):
            raise ValueError(f'Expected a recipe list from {url}')
        if not batch:
            break
        for recipe in batch:
            if not isinstance(recipe, dict) or not isinstance(recipe.get('id'), int):
                raise ValueError(f'Invalid recipe in page {page}')
            if recipe['id'] in seen:
                raise ValueError(f'Repeated recipe ID {recipe["id"]}; check API pagination')
            seen.add(recipe['id'])
            recipes.append(recipe)
        print(f'Fetched page {page}: {len(batch)} recipes', flush=True)
        page += 1
    return recipes if limit is None else recipes[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='Fetch a small sample; default is the entire API catalog')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    # Complete downloads before replacing existing outputs. Network failures exit visibly.
    recipes = download(args.limit)
    if not recipes:
        raise ValueError('API returned no recipes; existing outputs were not changed')
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = [{'id': recipe['id'], 'source_url': f'{API}/{recipe["id"]}',
                'fetched_at': fetched_at, 'recipe': recipe, 'audit': audit(recipe)}
               for recipe in recipes]
    available = Counter(key for row in records for key in row['audit']['available_fields'])
    report = {'source': API, 'fetched_at': fetched_at, 'recipes': len(records),
              'requested_limit': args.limit, 'field_coverage': dict(available),
              'instruction_review_required': len(records), 'training_ready': 0,
              'note': 'Field presence is not proof of a complete or validated brewing procedure. '
                      'Review original instructions and source reuse terms before training.'}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'raw.json').write_text(json.dumps(recipes, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUTPUT / 'recipes.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
