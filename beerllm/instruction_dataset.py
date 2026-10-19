"""Build a small source-checked teaching experiment without changing the collection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from .common import ROOT, read_jsonl, write_json, write_jsonl
from .curate import SYSTEM

GENERAL_URL = 'https://homebrewersassociation.org/tutorials/all-grain-batch-sparge-homebrewing/all-grain-batch-sparge-homebrewing/'
GENERAL_CHECKS = ('General brewing checks, separate from the recipe: clean equipment first, then sanitize '
                  'anything touching the cooled wort. Check fermentation with gravity readings that stay '
                  'consistent over a few days; a quiet airlock or a calendar date alone is not proof it is finished.')
NON_INGREDIENTS = {'additional ingredients','other fermentables','malts & adjuncts','water profile'}


def source_notes(recipe):
    ingredients='\n'.join('- '+x['source_text'] for x in recipe['ingredients']
                           if x['source_text'].casefold() not in NON_INGREDIENTS)
    return (f"Recipe: {recipe['title']}\nBatch: {recipe['batch_yield_original']}\nIngredients:\n{ingredients}"
            '\nSource directions:\n'+'\n'.join(recipe['directions']))


def guide(recipe,steps=None,gaps=None):
    ingredients='\n'.join('- '+x['source_text'] for x in recipe['ingredients']
                           if x['source_text'].casefold() not in NON_INGREDIENTS)
    answer=(f"Let's make {recipe['title']}. The source batch is {recipe['batch_yield_original']}."
            f'\n\nIngredients\n{ingredients}\n\nWhat to do\n')
    answer+='\n\n'.join(f'{i}. {step}' for i,step in enumerate(steps or recipe['directions'],1))
    if gaps: answer+='\n\nBefore you brew: '+gaps
    return answer+'\n\n'+GENERAL_CHECKS


def row(recipe,task,question,answer):
    return {'recipe_id':recipe['recipe_id'],'family_id':recipe['family_id'],
            'split':recipe['split'],'task':task,'source_url':recipe['source_url'],
            'source_html_sha256':recipe['provenance'].get('html_sha256'),
            'review_status':'source_checked_experiment_not_brewer_reviewed',
            'general_reference':GENERAL_URL,
            'messages':[{'role':'system','content':SYSTEM},
                        {'role':'user','content':question},
                        {'role':'assistant','content':answer}]}


def lesson_rows(recipe,lesson):
    if recipe['split']!='train' or recipe['audit']['tier']!='instruction_candidate':
        raise ValueError('Teaching lessons must use training-partition instruction candidates')
    answer=guide(recipe,lesson['steps'],lesson['gaps'])
    result=[row(recipe,'named_recipe',f"Teach me to brew {recipe['title']}. Include the ingredients, steps and things I should watch out for.",answer),
            row(recipe,'supplied_recipe','Explain these recipe notes to a beginner. Preserve the quantities and do not fill gaps with guessed numbers.\n\n'+source_notes(recipe),answer)]
    result.extend(row(recipe,'followup',q,a) for q,a in lesson['followups'])
    return result


def build(input_path,lessons_path,output):
    recipes=read_jsonl(input_path)
    lessons=json.loads(lessons_path.read_text())
    signatures={'source_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
                'lessons_sha256':hashlib.sha256(lessons_path.read_bytes()).hexdigest(),
                'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if any(previous.get(k)!=v for k,v in signatures.items()):
            raise ValueError('Dataset inputs or builder changed; choose a new output directory')
    train=[];selected=[]
    for lesson in lessons['lessons']:
        matches=[r for r in recipes if r['source_url'].rstrip('/').endswith('/'+lesson['slug'])
                 or (lesson.get('title_contains') and lesson['title_contains'] in r['title'])]
        if len(matches)!=1: raise ValueError(f"Expected one recipe for {lesson['slug']}, got {len(matches)}")
        r=matches[0]
        if r['family_id'] in {x['family_id'] for x in selected}: raise ValueError('Duplicate teaching family')
        train.extend(lesson_rows(r,lesson));selected.append(r)
    general={'recipe_id':'general-process','family_id':'general-process','split':'train',
             'provenance':{},'source_url':GENERAL_URL}
    train.extend([
        row(general,'general_check','My airlock stopped bubbling. Can I bottle now?',
            'Not from that observation alone. Take gravity readings over a few days to check that they stay consistent and fermentation has finished. A quiet airlock can be misleading. Do not bottle just because a fixed number of days has passed.'),
        row(general,'general_check','Why must I sanitize equipment after the boil if I already cleaned it?',
            'Cleaning removes debris. Sanitation is a separate step to reduce contamination of the cooled wort. Clean first, then sanitize the fermenter and anything else that will touch the wort after boiling. Do not assume that visibly clean equipment is already sanitized.')])
    # Keep all recipes from the pre-existing validation/test families out of training.
    heldout={name:[r for r in recipes if r['split']==name and r['audit']['tier']=='instruction_candidate']
             for name in ['validation','test']}
    for name,rs in heldout.items():
        examples=[row(r,'supplied_recipe','Explain these source recipe notes as ingredients and ordered brewing steps. Acknowledge missing details.\n\n'+source_notes(r),
                      guide(r,gaps='These are source instructions, not a verified complete brew sheet. Confirm any missing recipe parameters.')) for r in rs]
        write_jsonl(output/f'{name}.jsonl',examples)
    train.sort(key=lambda r:hashlib.sha256((r['recipe_id']+r['messages'][1]['content']).encode()).hexdigest())
    write_jsonl(output/'train.jsonl',train)
    write_jsonl(output/'selected_recipes.jsonl',selected)
    manifest=signatures | {'train_examples':len(train),'training_recipe_families':len(selected),
                           'train_tasks':dict(Counter(r['task'] for r in train)),
                           'validation_examples':len(heldout['validation']),'test_examples':len(heldout['test']),
                           'expert_reviewed':False,'status':'user_authorized_learning_experiment',
                           'note':'Handwritten teaching plans use eight source-checked training recipes. Held-out references remain unreviewed source-text drafts; neither loss nor pattern checks establish brewing accuracy.'}
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'data/curated_v1/recipes.jsonl')
    p.add_argument('--lessons',type=Path,default=ROOT/'configs/instruction-lessons.json')
    p.add_argument('--output',type=Path,default=ROOT/'data/instruction_v1')
    args=p.parse_args()
    print(json.dumps(build(args.input,args.lessons,args.output),indent=2))


if __name__=='__main__': main()
