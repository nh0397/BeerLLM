"""Build a frozen, varied teaching pilot and separate development/final suites."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from .aha import parse_recipe
from .common import ROOT, read_jsonl, write_json, write_jsonl
from .curate import SYSTEM, repair_ingredients, text
from .instruction_dataset import GENERAL_URL, guide, lesson_rows, row, source_notes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(recipe):
    """Verify saved HTML and re-extraction; this is not semantic or expert review."""
    path=ROOT/recipe['provenance']['raw_html']
    html=gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text()
    if hashlib.sha256(html.encode()).hexdigest()!=recipe['provenance']['html_sha256']:
        raise ValueError('Source HTML hash differs: '+recipe['title'])
    original=parse_recipe(html,recipe['source_url'])
    ingredients,_=repair_ingredients(original['ingredients_text'])
    if ([x['source_text'] for x in ingredients]!=[x['source_text'] for x in recipe['ingredients']]
            or [text(x) for x in original['directions_text']]!=recipe['directions']):
        raise ValueError('Re-extracted source differs: '+recipe['title'])
    return {'title':recipe['title'],'html_sha256':recipe['provenance']['html_sha256'],
            'ingredients_and_directions_match':True}


def expanded_rows(recipe,lesson,variants):
    examples=lesson_rows(recipe,lesson)
    if len(variants)!=len(lesson['followups']): raise ValueError('Missing follow-up variants')
    for (question,answer),questions in zip(lesson['followups'],variants):
        if len(questions)!=2 or len(set([question,*questions]))!=3:
            raise ValueError('Each follow-up requires two distinct wordings')
        examples.extend(row(recipe,'followup_variant',q,answer) for q in questions)
        conversation=row(recipe,'conversation_followup',questions[0],answer)
        conversation['messages'][1:1]=[
            {'role':'user','content':f"Walk me through {recipe['title']} as a beginner."},
            {'role':'assistant','content':guide(recipe,lesson['steps'],lesson['gaps'])}]
        examples.append(conversation)
    ingredients='\n'.join(x['source_text'] for x in recipe['ingredients'])
    examples.append(row(recipe,'ingredient_list',f"What ingredients and batch size are listed for {recipe['title']}?",
                        f"The source batch is {recipe['batch_yield_original']}.\n"+ingredients))
    examples.append(row(recipe,'source_gaps',f"What should I confirm before brewing {recipe['title']}?",
                        lesson['gaps']))
    return examples


def ensure_disjoint(training,heldout):
    if {r['family_id'] for r in training}&{r['family_id'] for r in heldout}:
        raise ValueError('Recipe family leaks into training')


def case(identifier,category,recipe,question,answer,max_tokens=220,context=False):
    if context: question+='\n\nSource recipe notes:\n'+source_notes(recipe)
    return {'case_id':identifier,'category':category,'source_url':recipe['source_url'] if recipe else GENERAL_URL,
            'recipe_id':recipe['recipe_id'] if recipe else None,'family_id':recipe['family_id'] if recipe else None,
            'max_new_tokens':max_tokens,'checks':[],
            'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':question},
                        {'role':'assistant','content':answer}]}


def build(recipes_path,config_path,output,verify=True):
    config=json.loads(config_path.read_text())
    base_path=ROOT/config['base_lessons']
    base=json.loads(base_path.read_text())['lessons']
    recipes=read_jsonl(recipes_path)
    lessons=base+config['extra_lessons']
    variants={x['slug']:x['questions'] for x in config['variants']+config['extra_lessons']}
    def find(lesson):
        matches=[r for r in recipes if r['source_url'].rstrip('/').endswith('/'+lesson['slug'])
                 or (lesson.get('title_contains') and lesson['title_contains'] in r['title'])]
        if len(matches)!=1: raise ValueError('Recipe selection must be unambiguous')
        return matches[0]
    selected=[find(x) for x in lessons]
    if len({r['family_id'] for r in selected})!=len(selected): raise ValueError('Duplicate teaching family')
    final=[next(r for r in recipes if r['title']==title) for title in config['reserved_final_titles']]
    validation=[r for r in recipes if r['split']=='validation' and r['audit']['tier']=='instruction_candidate']
    ensure_disjoint(selected,final+validation)
    source_checks=[verify_source(r) for r in selected+validation+final] if verify else []
    training=[]
    for recipe,lesson in zip(selected,lessons): training.extend(expanded_rows(recipe,lesson,variants[lesson['slug']]))
    general={'recipe_id':'general-process','family_id':'general-process','split':'train',
             'provenance':{},'source_url':GENERAL_URL}
    general_pairs=[
        ('Can I bottle as soon as my airlock becomes quiet?',
         'A quiet airlock alone does not establish completion. Check gravity repeatedly over a few days and confirm that it stays consistent before deciding fermentation is finished. Do not bottle solely because bubbling has stopped.'),
        ('The recipe says fourteen days. Is the calendar enough to tell me to bottle?',
         'No. Use gravity readings that stay consistent over a few days to check completion. A fixed time estimate alone is not proof that fermentation has finished.'),
        ('Why sanitize my fermenter after cleaning it?',
         'Cleaning removes debris; sanitation is a separate step to reduce contamination. Clean first, then sanitize the fermenter and everything touching cooled wort.'),
        ('What does pitching mean, and does a mash temperature tell me when to pitch?',
         'Pitching means adding yeast to cooled wort. A mash temperature belongs to the grain-and-water rest, not yeast pitching. Use the recipe and the chosen yeast guidance for pitching; do not repurpose a mash temperature.')]
    training.extend(row(general,'general_check',q,a) for q,a in general_pairs)
    for title in ['My Secret Amber 17','Unshared Garage Lager']:
        training.append(row(general,'missing_recipe',f"Give exact ingredients and fermentation settings for my private recipe {title}. I have not supplied the recipe.",
            'I do not have that private recipe, so I cannot know its exact ingredients or fermentation settings. Share its batch size, ingredient list and process notes, and I can explain them. I will not invent those recipe details.'))
    training.sort(key=lambda r:hashlib.sha256(json.dumps(r['messages'],ensure_ascii=False).encode()).hexdigest())
    by_slug={lesson['slug']:(r,lesson) for r,lesson in zip(selected,lessons)}
    dev=[]
    for identifier,slug,index,question in [
        ('volume-paraphrase','beer-recipe-of-the-week-dusseldorf-altbier',0,'For Düsseldorf Altbier, tell me how much wort is boiled and how much ends up in the fermenter. Include both units.'),
        ('lager-stages-paraphrase','helicious-helles',0,'List Helicious Helles primary fermentation and later cold storage as two stages, with the temperature and duration of each.'),
        ('smash-strike-paraphrase','your-first-smash-ale',1,'For the one-gallon SMaSH recipe, distinguish the initial water heating from the grain rest and the mash-out.'),
        ('fullers-dryhop-paraphrase','fullers-chiswick-bitter-clone',0,"For Fuller's Chiswick Bitter, explain where each of the three East Kent Golding dry-hop doses belongs, including the bottling alternative.")]:
        r,lesson=by_slug[slug]
        dev.append(case(identifier,'seen_fact_new_wording',r,question,lesson['followups'][index][1]))
    r,lesson=by_slug['your-first-smash-ale']
    dev.append(case('smash-full-guide','seen_recipe_guide',r,'Teach me the complete one-gallon SMaSH brew, with ingredients, stages and missing settings.',guide(r,lesson['steps'],lesson['gaps']),1000))
    for r in validation[:1]:
        dev.append(case('validation-supplied-guide','validation_supplied_recipe',r,
            'Explain this source as a beginner brew guide. Preserve the ingredients and process; flag missing settings.',
            guide(r,gaps='Confirm missing recipe-specific settings instead of guessing.'),1000,True))
    dev.append(case('unshared-recipe','uncertainty',None,'Give exact grain weights and fermentation temperatures for my unpublished recipe Copper Lantern 426. You have never seen its notes.',
        'I do not have that unpublished recipe. Please share its batch size, ingredients and process notes; I cannot know its exact weights or fermentation temperatures.',180))
    dev.append(case('gravity-check','general_process',None,'The calendar says fourteen days and the bubbles have stopped. Explain how I should decide whether to package.',
        'Check gravity with repeated readings over a few days and confirm that it stays consistent. Neither the calendar nor a quiet airlock alone establishes completion. Do not bottle solely on that basis.',180))
    final_cases=[]
    for r in final:
        final_cases.append(case('final-'+r['recipe_id'][:12],'fresh_family_supplied_recipe',r,
            'Explain the complete recipe to a beginner, with ingredients and ordered steps. Keep different volumes and temperature stages separate, and identify missing instructions.',
            guide(r,gaps='The source does not establish every setting; identify omissions rather than inventing details.'),1100,True))
    train_questions={r['messages'][-2]['content'] for r in training}
    if any(r['messages'][1]['content'] in train_questions for r in dev): raise ValueError('Development prompt is in training')
    ensure_disjoint(training,validation+final)
    exact=next(r for r in training if r['task']=='followup' and
               r['messages'][1]['content']=='Is Düsseldorf Altbier a three-gallon batch?')
    probes=[{'case_id':'exact-volume','category':'training_fit','messages':exact['messages'],'max_new_tokens':140},
            dict(dev[0],max_new_tokens=140),dict(dev[1],max_new_tokens=160)]
    datasets={'train.jsonl':training,
              'validation.jsonl':[row(r,'supplied_recipe','Explain these notes, preserving quantities and acknowledging missing settings.\n\n'+source_notes(r),guide(r,gaps='Confirm missing parameters before brewing.')) for r in validation],
              'development.jsonl':dev,'final.jsonl':final_cases,'probes.jsonl':probes,'selected_recipes.jsonl':selected}
    manifest={'source_sha256':sha(recipes_path),'config_sha256':sha(config_path),'base_lessons_sha256':sha(base_path),
              'builder_sha256':sha(Path(__file__)),'train_examples':len(training),'train_families':len(selected),
              'tasks':dict(Counter(r['task'] for r in training)),'validation_families':len(validation),
              'final_families':len(final),'final_titles':config['reserved_final_titles'],'expert_reviewed':False,
              'note':'Question variants and conversations reuse facts; example count is not independent recipe count. Final families are excluded from this experiment and earlier authored teaching data. Development results may inform choices.',
              'source_checks':source_checks}
    path=output/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text())
        file_hashes=previous.pop('files_sha256')
        if previous!=manifest: raise ValueError('Frozen inputs differ; choose a new output directory')
        if any(read_jsonl(output/name)!=rows for name,rows in datasets.items()): raise ValueError('Frozen dataset was modified')
        if any(sha(output/name)!=digest for name,digest in file_hashes.items()): raise ValueError('Frozen file hash differs')
        return dict(manifest,files_sha256=file_hashes)
    for name,rows in datasets.items(): write_jsonl(output/name,rows)
    manifest['files_sha256']={name:sha(output/name) for name in datasets}
    # Include file hashes in both first-build and repeat-build comparisons.
    write_json(path,manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes',type=Path,default=ROOT/'data/curated_v1/recipes.jsonl')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/instruction-v2-lessons.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/instruction_v2')
    args=parser.parse_args()
    print(json.dumps(build(args.recipes,args.config,args.output),indent=2))


if __name__=='__main__': main()
