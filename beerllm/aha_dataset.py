"""Convert extracted AHA recipes to natural-language fine-tuning candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from .common import ROOT, read_jsonl, write_json, write_jsonl

SYSTEM = ('You are a friendly brewing teacher. Explain recipes in clear, natural language for a beginner. '
          'Give ingredients and ordered actions, preserve quantities and units, and explain why or what to avoid '
          'when the source supports it. Do not invent missing recipe instructions.')


def family_key(title):
    text = re.sub(r'\b(?:v\d+|20\d\d|all.grain|extract)\b', '', title.casefold())
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def conversation(record):
    """Keep source facts/directions; this formats text but does not author new explanations."""
    title = record['title']
    answer = f"Here's the recipe for {title}."
    if record.get('batch_yield'):
        answer += f" It makes {record['batch_yield']}."
    answer += "\n\nWhat you'll need:\n" + '\n'.join('- '+x for x in record['ingredients_text'])
    answer += "\n\nHow to brew it:\n" + '\n\n'.join(
        f'{i}. {step}' for i, step in enumerate(record['directions_text'], 1))
    missing = record['audit']['missing_stage_mentions']
    if missing:
        answer += '\n\nThe extracted directions need further review for: ' + ', '.join(missing).replace('_',' ') + '.'
    return {'source_url':record['source_url'], 'source_html_sha256':record['html_sha256'],
            'family_id':family_key(title), 'status':'candidate_needs_review',
            'messages':[{'role':'system','content':SYSTEM},
                        {'role':'user','content':f"I'd like to brew {title}. Please give me the ingredients and walk me through the brewing instructions in plain language."},
                        {'role':'assistant','content':answer}]}


def build(records):
    unique = {}
    for r in records:
        if not r.get('ingredients_text') or not r.get('directions_text'):
            raise ValueError('Recipes must contain original ingredients and directions')
        if r['source_url'] in unique and unique[r['source_url']]['html_sha256'] != r['html_sha256']:
            raise ValueError('Conflicting versions of a source recipe; review before export')
        unique[r['source_url']] = r
    groups = {}
    for r in unique.values():
        groups.setdefault(family_key(r['title']), []).append(r)
    if len(groups)<3:
        raise ValueError('Need at least three distinct recipe families for separate train, validation and test splits')
    # Deterministic grouping and ordering for a pilot. Additions can move splits: freeze a release before training.
    ordered = sorted(groups, key=lambda key:hashlib.sha256(('aha-v1:'+key).encode()).hexdigest())
    holdout = max(1,len(ordered)//10)
    assignments = {key:('test' if i<holdout else 'validation' if i<2*holdout else 'train')
                   for i,key in enumerate(ordered)}
    splits = {key:[] for key in ['train','validation','test']}
    for family in ordered:
        for r in groups[family]:
            row=conversation(r)
            row['split']=assignments[family]
            splits[row['split']].append(row)
    return splits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/aha/recipes.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'data/aha/training_candidates')
    args=parser.parse_args()
    splits=build(read_jsonl(args.input))
    for split,rows in splits.items():
        write_jsonl(args.output/f'{split}.jsonl',rows)
    report={'examples':{split:len(rows) for split,rows in splits.items()},
            'format':'messages: system, user, assistant',
            'prompts_include_recipe_json':False,
            'status':'format_valid_candidates_not_reviewed_training_release',
            'expert_reviewed':False,
            'limitations':['Original instructions are formatted, not independently verified or expanded with new explanations.',
                           'Near duplicates across different titles still need review.',
                           'Adding recipes can change split assignment; freeze a release before training.',
                           'A few records demonstrate format compatibility only; do not establish model quality.']}
    write_json(args.output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
