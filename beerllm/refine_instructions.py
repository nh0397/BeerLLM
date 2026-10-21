"""Refine the frozen teaching pilot with direct answers and source-specific Q&A."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from .common import ROOT, read_jsonl, write_json, write_jsonl
from .instruction_dataset import GENERAL_URL, row, source_notes
from .instruction_v2 import ensure_disjoint, sha

FILES=['train.jsonl','validation.jsonl','development.jsonl','final.jsonl','probes.jsonl','selected_recipes.jsonl']


def scoped(example,system):
    result=copy.deepcopy(example)
    if result['messages'][0]['role']!='system': raise ValueError('Expected an explicit system prompt')
    result['messages'][0]['content']=system
    return result


def build(source,config_path,output):
    upstream=json.loads((source/'manifest.json').read_text())
    for name in FILES:
        if upstream['files_sha256'][name]!=sha(source/name): raise ValueError('Upstream dataset changed: '+name)
    config=json.loads(config_path.read_text());system=config['system_prompt']
    original=read_jsonl(source/'train.jsonl');recipes=read_jsonl(source/'selected_recipes.jsonl')
    training=[scoped(x,system) for x in original]
    covered=set()
    for lesson in config['lessons']:
        matches=[r for r in recipes if r['source_url'].rstrip('/').endswith('/'+lesson['slug'])]
        if len(matches)!=1: raise ValueError('Unknown or ambiguous refinement recipe')
        recipe=matches[0]
        if recipe['split']!='train' or recipe['recipe_id'] in covered:
            raise ValueError('Refinements require distinct training recipes')
        covered.add(recipe['recipe_id'])
        for fact in lesson['facts']:
            targets=[r for r in original if r['recipe_id']==recipe['recipe_id'] and r['task']=='followup'
                     and r['messages'][1]['content']==fact['original_question']]
            if len(targets)!=1: raise ValueError('Expected one original training follow-up')
            old_answer=targets[0]['messages'][-1]['content']
            # A direct answer fits open questions better than a reused yes/no opening.
            for example in training:
                if (example['recipe_id']==recipe['recipe_id'] and example['task'] in ['followup_variant','conversation_followup']
                        and example['messages'][-1]['content']==old_answer):
                    example['messages'][-1]['content']=fact['answer']
                    example['review_status']='assistant_authored_source_checked_not_expert_reviewed'
            question=f"For {recipe['title']}, {fact['question']}"
            training.append(scoped(row(recipe,'direct_fact',question,fact['answer']),system))
            training.append(scoped(row(recipe,'source_fact','Use only these notes to answer: '+question+
                '\n\nSource recipe notes:\n'+source_notes(recipe),fact['answer']),system))
        ingredient=next(r for r in original if r['recipe_id']==recipe['recipe_id'] and r['task']=='ingredient_list')
        training.append(scoped(row(recipe,'ingredient_variant',f"List the source ingredients for {recipe['title']}, with amounts and batch size.",
            ingredient['messages'][-1]['content']),system))
    if covered!={r['recipe_id'] for r in recipes}: raise ValueError('Every teaching recipe needs a refinement')
    general={'recipe_id':'general-process','family_id':'general-process','split':'train','provenance':{},'source_url':GENERAL_URL}
    pairs=[
        ('Give exact grain weights and yeast for my unpublished recipe Night Mill 71. I have not shared its notes.',
         'I do not have that unpublished recipe. Please share its batch size, ingredient list and process notes. I cannot know its exact grain weights or yeast without those details.'),
        ('My private Backyard Pilsner is five gallons. Tell me its exact grain bill and yeast even though I have not supplied them.',
         'The five-gallon batch size is known, but it does not tell me the grain bill or yeast. Please share those recipe details. I cannot infer exact ingredient amounts from the name and batch size.'),
        ('Two weeks have passed and bubbling is quiet. How do I check whether my beer is ready for bottles?',
         'Take gravity readings over a few days and confirm that they stay consistent. A calendar estimate and quiet bubbling alone do not establish completion. Do not bottle solely on those observations.'),
        ('My hydrometer readings are still changing, but the airlock is quiet. Is that enough to bottle?',
         'A quiet airlock is not enough. Check for gravity readings that stay consistent over a few days and confirm fermentation has finished before bottling. Changing readings do not meet that stability check.')]
    training.extend(scoped(row(general,'general_check_refinement',q,a),system) for q,a in pairs)
    training.sort(key=lambda r:hashlib.sha256(json.dumps(r['messages'],ensure_ascii=False).encode()).hexdigest())
    datasets={name:[scoped(r,system) for r in read_jsonl(source/name)] for name in FILES if name not in ['train.jsonl','selected_recipes.jsonl']}
    datasets['selected_recipes.jsonl']=recipes;datasets['train.jsonl']=training
    ensure_disjoint(training,datasets['validation.jsonl']+datasets['final.jsonl'])
    questions={r['messages'][-2]['content'] for r in training}
    if any(r['messages'][1]['content'] in questions for r in datasets['development.jsonl']+datasets['final.jsonl']):
        raise ValueError('Evaluation question copied into training')
    manifest={'upstream_manifest_sha256':sha(source/'manifest.json'),
              'upstream_files_sha256':{name:sha(source/name) for name in FILES},'config_sha256':sha(config_path),
              'builder_sha256':sha(Path(__file__)),'system_prompt':system,'train_examples':len(training),
              'training_recipe_families':len(recipes),'tasks':dict(Counter(r['task'] for r in training)),
              'expert_reviewed':False,
              'note':'Same eleven source-checked teaching recipes. Direct answers, source-specific factual questions and scope-aware instructions are refinements, not new independent recipes. Final families remain excluded.'}
    path=output/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text());file_hashes=previous.pop('files_sha256')
        if previous!=manifest: raise ValueError('Frozen refinement inputs changed; choose a new output directory')
        if any(sha(output/name)!=digest or read_jsonl(output/name)!=datasets[name] for name,digest in file_hashes.items()):
            raise ValueError('Frozen refinement data changed')
        return dict(manifest,files_sha256=file_hashes)
    if output.exists() and any(output.iterdir()): raise ValueError('Incomplete release exists; preserve it and choose a new output directory')
    for name,rows in datasets.items(): write_jsonl(output/name,rows)
    manifest['files_sha256']={name:sha(output/name) for name in datasets};write_json(path,manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/instruction_v2')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/instruction-refinements.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/instruction_v3')
    args=parser.parse_args();print(json.dumps(build(args.source,args.config,args.output),indent=2))


if __name__=='__main__': main()
