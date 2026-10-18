"""Build source-grounded full recipe guides, with explicit gaps, not invented steps."""
import argparse
from collections import Counter
import hashlib
import json
import re
from .common import ROOT, SYSTEM, write_json, write_jsonl
from scripts.fetch_recipes import audit

PROCESS_SOURCE = 'https://brewdogmedia.s3.eu-west-2.amazonaws.com/docs/2019+DIY+DOG+-+V8.pdf#page=14'
FERMENTATION_SOURCE = 'https://homebrewersassociation.org/tutorials/all-grain-batch-sparge-homebrewing/all-grain-batch-sparge-homebrewing/'


def amount(value):
    return f"{value['value']:g} {value['unit']}"


def family_name(recipe):
    # Keep versions, collaborations and single-hop series together.
    name = recipe['name'].casefold()
    name = re.sub(r'\([^)]*\)|\b(?:v\d+(?:\.\d+)?|20\d\d|current|fanzine)\b', ' ', name)
    name = name.split(' - ')[0].split(':')[0]
    return re.sub(r'[^a-z0-9]+', ' ', name).strip()


def family_groups(recipes):
    """Union related names OR matching ingredient sets before assigning splits."""
    parent = {r['id']: r['id'] for r in recipes}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    keys = {}
    for r in recipes:
        ing = r['ingredients']
        names = tuple(sorted((kind, x['name'].casefold().strip())
                             for kind in ['malt','hops'] for x in ing.get(kind, [])))
        for key in [('name', family_name(r)), ('ingredients', names, ing.get('yeast','').casefold())]:
            if key in keys:
                a, b = root(r['id']), root(keys[key])
                parent[max(a,b)] = min(a,b)
            else:
                keys[key] = r['id']
    return {i: str(root(i)) for i in parent}


def compact_recipe(r):
    return {k: r.get(k) for k in ['id','name','volume','boil_volume','ingredients','method','brewers_tips']}


def guide(r):
    ing, method = r['ingredients'], r['method']
    lines = [f"{r['name']} — {amount(r['volume'])} batch.", '', 'Ingredients (source quantities):']
    lines += [f"- {x['name']}: {amount(x['amount'])}." for x in ing['malt']]
    lines += [f"- {x['name']}: {amount(x['amount'])}; hop addition label: {x['add']}." for x in ing['hops']]
    lines += [f"- Yeast: {ing['yeast']}.", '', 'Brewing procedure:',
              '1. Preparation: clean equipment and sanitize everything that contacts cooled wort. '
              'Measure the ingredients; calculate strike and sparge water for your equipment. '
              'Batch volume is not the total water requirement.',
              '2. Mash: mix the grain with strike water and maintain these source-specified rests: '
              + '; '.join(f"{amount(s['temp'])} for {s['duration']:g} minutes" for s in method['mash_temp']) + '.',
              '3. Separate the wort from the grain; rinse as appropriate for your system. '
              + (f"The source pre-boil volume is {amount(r['boil_volume'])}." if r.get('boil_volume') else 'Pre-boil volume is not specified.'),
              '4. Bring the wort to a rolling boil and follow the hop addition labels listed above. '
              'The API does not specify total boil duration. Do not convert start/middle/end labels into exact minutes without confirming the schedule. '
              'Dry-hop additions are separate from the boil.',
              '5. Cool the wort promptly to a suitable pitching temperature for the specified yeast, '
              'transfer with sanitized equipment, aerate as appropriate for the yeast, and pitch. '
              'Fit an airlock; maintain the source fermentation temperature of '
              + amount(method['fermentation']['temp']) + '.',
              '6. Follow any recipe-specific notes below. Check gravity over several days to assess fermentation completion; '
              'a fixed number of days or lack of bubbles alone is insufficient.',
              '7. Condition and package only after fermentation is complete. Choose a verified bottling or kegging procedure '
              'for the actual beer volume and carbonation target. This record does not establish a priming dose or a full packaging schedule.']
    if method.get('twist'):
        lines += ['', 'Source special instructions: ' + method['twist']]
    if r.get('brewers_tips'):
        lines += ['', 'Source brewer tips: ' + r['brewers_tips']]
    lines += ['', 'Scope: ingredients, mash rests and fermentation temperature come from the recipe. '
              'Cleaning, wort handling and completion checks are general brewing guidance. '
              'Confirm boil timing, dry-hop contact time where relevant, conditioning and packaging details before brewing; '
              'this is not a fully verified brew sheet.']
    return '\n'.join(lines)


def build(recipes):
    accepted, rejected = [], []
    required = {'batch_volume','malt_amounts','hop_amounts','hop_addition_labels','yeast','mash_schedule','fermentation_temperature'}
    seen = set()
    for r in recipes:
        if r['id'] in seen:
            raise ValueError(f"Duplicate recipe ID {r['id']}")
        seen.add(r['id'])
        missing = required.intersection(audit(r)['missing_fields'])
        # Keep advanced special processes for later expert curation.
        text = ' '.join(str(r.get(k, '')) for k in ['name','description']) + ' ' + str(r.get('method',{}).get('twist') or '')
        reasons = sorted(missing)
        if (r.get('abv') or 0) > 12 or re.search(r'barrel|sour|brett|lacto|freeze|distill|mixed.culture', text, re.I):
            reasons.append('special_process_requires_review')
        for kind in ['malt', 'hops']:
            for x in r.get('ingredients',{}).get(kind,[]):
                q = x.get('amount') or {}
                if not isinstance(q.get('value'), (int,float)) or q['value'] <= 0 or q.get('unit') not in ['grams','kilograms']:
                    reasons.append('invalid_ingredient_quantity')
        if reasons:
            rejected.append({'recipe_id':r['id'], 'name':r['name'], 'reasons':sorted(set(reasons))})
        else:
            accepted.append(r)
    families = family_groups(accepted)
    splits = {s: [] for s in ['train','validation','test']}
    guides = []
    for r in accepted:
        family = families[r['id']]
        bucket = int(hashlib.sha256(('beerllm-v1:'+family).encode()).hexdigest()[:8],16) % 100
        split = 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'
        answer = guide(r)
        source = f"https://punkapi.dev/v3/beers/{r['id']}"
        guides.append({'recipe_id':r['id'],'split':split,'family_id':family,'source_url':source,'guide':answer,
                       'status':'programmatic_guide_with_explicit_gaps','expert_reviewed':False})
        for task in ['grounded_guide','named_recipe']:
            # Unseen recipe recall is not a fair held-out factual benchmark.
            if task == 'named_recipe' and split != 'train':
                continue
            prompt = f"Give me the ingredients and brewing procedure for {r['name']} at its original batch size. Identify any details that still need confirmation."
            if task == 'grounded_guide':
                prompt += '\nRecipe record:\n' + json.dumps(compact_recipe(r),ensure_ascii=False,separators=(',',':'))
            splits[split].append({'recipe_id':r['id'],'family_id':family,'task':task,'source_url':source,
                                  'process_sources':[PROCESS_SOURCE,FERMENTATION_SOURCE],
                                  'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt},
                                              {'role':'assistant','content':answer}]})
    return splits, guides, rejected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=__import__('pathlib').Path,default=ROOT/'data/punkapi/raw.json')
    args=parser.parse_args()
    splits,guides,rejected=build(json.loads(args.input.read_text(encoding='utf-8')))
    out=ROOT/'data/training'
    for split,rows in splits.items():
        if not rows:
            raise ValueError(f'No {split} examples; add more source recipes')
        write_jsonl(out/f'{split}.jsonl',rows)
    write_jsonl(out/'guides.jsonl',guides)
    write_jsonl(out/'rejected.jsonl',rejected)
    report={'accepted_recipes':len(guides),'rejected_recipes':len(rejected),
            'examples':{s:len(rows) for s,rows in splits.items()},
            'recipe_splits':dict(Counter(g['split'] for g in guides)),
            'expert_reviewed_recipes':0,
            'purpose':'Experimental full-guide fine-tuning with explicit unresolved steps; not validated complete brew sheets.',
            'evaluation':'Held-out recipes are supplied as context. Named recipe prompts train recall but are not used to claim unseen factual recall.'}
    write_json(out/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
