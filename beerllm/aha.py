"""Import original ingredients and written directions from selected public AHA pages."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from .common import ROOT, write_json, write_jsonl

ORIGIN = 'https://homebrewersassociation.org'
AGENT = 'BeerLLM-Learning/0.1'


def recipe_url(url):
    parsed = urlparse(url.strip())
    if (parsed.scheme != 'https' or parsed.hostname not in
            ['homebrewersassociation.org', 'www.homebrewersassociation.org']
            or not re.fullmatch(r'/homebrew-recipe/[^/]+/?', parsed.path)
            or parsed.username or parsed.password or parsed.port):
        raise ValueError('Expected a public HTTPS AHA /homebrew-recipe/<name>/ URL')
    return urlunparse(('https', 'homebrewersassociation.org', parsed.path.rstrip('/')+'/', '', '', ''))


def parse_recipe(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    for node in soup.select('script, style, nav, footer'):
        node.decompose()
    title = soup.select_one('h1')
    ingredients = soup.select_one('.ingredients')
    instructions = soup.select_one('[itemprop="recipeInstructions"]')
    if not title or not ingredients or not instructions:
        raise ValueError('Original ingredients/directions unavailable; page may be restricted or unsupported')
    # Specifications share the outer .ingredients container; isolate the ingredient list.
    ingredient_list = ingredients.select_one('ul, ol')
    if not ingredient_list:
        raise ValueError('Ingredient list not found')
    lines = [x.get_text(' ', strip=True) for x in ingredient_list.find_all('li')]
    directions = [x.get_text(' ', strip=True) for x in instructions.find_all(['p', 'li'])
                  if not x.find(['p', 'li'])]
    lines = [x for x in lines if x]
    directions = [x for x in directions if x]
    if not lines or not directions:
        raise ValueError('Empty ingredients or written directions')
    specs = {}
    for node in ingredients.select('p'):
        text = node.get_text(' ', strip=True)
        key, sep, value = text.partition(':')
        if sep:
            specs[key.strip()] = value.strip()
    yield_node = soup.select_one('[itemprop="recipeYield"]')
    batch = specs.get('Yield') or (yield_node.get_text(' ', strip=True) if yield_node else None)
    style = soup.select_one('[itemprop="recipeCuisine"]')
    description = soup.select_one('.recipe-description')
    text = ' '.join(directions).lower()
    stage_terms = {'mash_or_steep': r'\bmash|\bsteep', 'boil': r'\bboil',
                   'cooling': r'\bcool|\bchill', 'fermentation': r'\bferment|\bpitch',
                   'packaging': r'\bbottl|\bkeg|\bpackag'}
    stages = {key:bool(re.search(pattern,text)) for key,pattern in stage_terms.items()}
    return {'title':title.get_text(' ', strip=True), 'source_url':url,
            'style':style.get_text(' ', strip=True) if style else None,
            'batch_yield':batch, 'specifications':specs, 'ingredients_text':lines,
            'directions_text':directions,
            'description':description.get_text(' ',strip=True) if description else None,
            'audit':{'stage_mentions':stages, 'missing_stage_mentions':[key for key,present in stages.items() if not present],
                     'batch_yield_present':bool(batch), 'status':'needs_review',
                     'note':'Stage mentions do not prove complete or correct instructions; review timings, temperatures and completion criteria.'},
            'rights_status':'unverified', 'expert_reviewed':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('urls', nargs='*', help='Selected public AHA recipe URLs')
    parser.add_argument('--urls-file', type=Path, help='One selected public recipe URL per line')
    args = parser.parse_args()
    selected = list(args.urls)
    if args.urls_file:
        selected += [line.strip() for line in args.urls_file.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
    if not selected:
        parser.error('Provide recipe URLs or --urls-file')
    urls = list(dict.fromkeys(recipe_url(url) for url in selected))
    out = ROOT/'data/aha'
    raw = out/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = AGENT
    robots_response = session.get(ORIGIN+'/robots.txt', timeout=30)
    robots_response.raise_for_status()
    robots = RobotFileParser()
    robots.parse(robots_response.text.splitlines())
    delay = max(10, robots.crawl_delay(AGENT) or robots.crawl_delay('*') or 0)
    last_request = time.monotonic()
    records, errors = [], []
    for url in urls:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        html_path, metadata_path = raw/f'{key}.html', raw/f'{key}.json'
        try:
            if not robots.can_fetch(AGENT, url):
                raise ValueError('robots.txt disallows this URL')
            if html_path.exists():
                html = html_path.read_text(encoding='utf-8')
            else:
                time.sleep(max(0, delay-(time.monotonic()-last_request)))
                response = session.get(url, timeout=30, allow_redirects=False)
                last_request = time.monotonic()
                response.raise_for_status()
                if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type',''):
                    raise ValueError('Expected an accessible recipe HTML page; redirects are not followed')
                html = response.text
                html_path.write_text(html,encoding='utf-8')
                write_json(metadata_path,{'source_url':url,'fetched_at':datetime.now(timezone.utc).isoformat()})
            record = parse_recipe(html,url)
            record['fetched_at'] = (json.loads(metadata_path.read_text())['fetched_at']
                                    if metadata_path.exists() else None)
            record['html_sha256'] = hashlib.sha256(html.encode()).hexdigest()
            record['raw_html'] = str(html_path.relative_to(ROOT))
            records.append(record)
            print('Imported:',record['title'],flush=True)
        except (requests.RequestException, ValueError) as exc:
            errors.append({'source_url':url,'error':str(exc)})
    # Each invocation is a batch. Cached originals are retained; prior batch outputs are replaced.
    write_jsonl(out/'recipes.jsonl',records)
    report = {'requested':len(urls),'imported':len(records),'errors':errors,
              'instructions':'Original source paragraphs, preserved in order; no generated steps.',
              'training_ready':0,'status':'candidate_records_for_review'}
    write_json(out/'report.json',report)
    print(f"Imported {len(records)} of {len(urls)} recipes. Review data/aha/report.json.")
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
