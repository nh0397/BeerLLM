"""Freeze a one-recipe teaching experiment and test actual generated chat history."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from .common import ROOT, GREEDY_DEFAULTS, generate, load_model, model_revision, read_jsonl, write_json, write_jsonl
from .chat import fit_history
from .instruction_dataset import guide
from .instruction_v2 import verify_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(recipes, lesson_path, output, verify=True):
    lesson=json.loads(lesson_path.read_text())
    matches=[r for r in read_jsonl(recipes) if r['title']==lesson['title']]
    if len(matches)!=1: raise ValueError('Expected one unambiguous recipe')
    recipe=matches[0]
    if recipe['split']!='train' or recipe['audit']['tier']!='instruction_candidate':
        raise ValueError('Focus recipe must be a source-checked training candidate')
    if recipe['provenance']['html_sha256']!=lesson['source_html_sha256']:
        raise ValueError('Authored lesson source hash differs')
    check=verify_source(recipe) if verify else None
    answers={'guide':guide(recipe,lesson['steps'],lesson['gaps']),
             'ingredients':f"The source batch is {recipe['batch_yield_original']}.\n"+
                           '\n'.join('- '+r['source_text'] for r in recipe['ingredients'])}
    for item in lesson['followups']:
        if item['key']!='ingredients': answers[item['key']]=item['answer']
    answers.update(lesson.get('extra_answers',{}))
    def messages(question,answer):
        return [{'role':'system','content':lesson['system_prompt']},
                {'role':'user','content':question}, {'role':'assistant','content':answers[answer]}]
    def row(question,answer,task):
        return {'recipe_id':recipe['recipe_id'],'family_id':recipe['family_id'],'split':'train',
                'task':task,'source_url':recipe['source_url'],
                'source_html_sha256':lesson['source_html_sha256'],
                'review_status':'source_checked_not_expert_reviewed','messages':messages(question,answer)}
    train=[row(q,'guide','complete_guide') for q in lesson['guide_questions']]
    train.extend(row(x['question'],x['key'],'followup') for x in lesson['followups'])
    for item in lesson['conversations']:
        r=row(item['question'],item['answer'],'conversation_followup')
        r['messages'][1:1]=[{'role':'user','content':item['prior_question']},
                            {'role':'assistant','content':answers[item['prior_answer']]}]
        train.append(r)
    def suite(items):
        return [{'case_id':x['id'],'category':x.get('category','recipe_followup'),
                 'recipe_id':recipe['recipe_id'],'family_id':recipe['family_id'],
                 'source_url':recipe['source_url'],'checks':[],
                 'max_new_tokens':x['max_new_tokens'],'messages':messages(x['question'],x['answer'])}
                for x in items]
    dev=suite(lesson['development']);final=suite(lesson['final'])
    dialogue={'scope':'same trained recipe; fresh wording and actual generated history, not unseen-recipe generalization',
              'system_prompt':lesson['system_prompt'],
              'turns':[dict(x,reference=answers[x['answer']]) for x in lesson['dialogue']]}
    training_questions={m['content'] for r in train for m in r['messages'] if m['role']=='user'}
    heldout_questions=[x['messages'][1]['content'] for x in dev+final]+[x['question'] for x in dialogue['turns']]+[lesson['validation_question']]
    if len(set(heldout_questions))!=len(heldout_questions) or training_questions&set(heldout_questions):
        raise ValueError('Evaluation question duplicated or copied into training')
    probes=[{'case_id':'exact-complete-guide','category':'training_fit','messages':train[0]['messages'],'max_new_tokens':1100},
            {'case_id':'exact-hop-timer','category':'training_fit','messages':messages(lesson['followups'][2]['question'],'hops'),'max_new_tokens':180}]
    datasets={'train.jsonl':train,'validation.jsonl':[row(lesson['validation_question'],'guide','validation_wording')],
              'development.jsonl':dev,'final.jsonl':final,'probes.jsonl':probes,'recipe.jsonl':[recipe]}
    # Validation repeats recipe facts under different wording; it is not an unfamiliar family.
    datasets['validation.jsonl'][0]['split']='validation'
    manifest={'scope':'one trained recipe; fitting and wording tests, not general brewing accuracy',
              'title':recipe['title'],'recipe_id':recipe['recipe_id'],'family_id':recipe['family_id'],
              'source_url':recipe['source_url'],'source_check':check,'expert_reviewed':False,
              'lesson_sha256':sha(lesson_path),'builder_sha256':sha(Path(__file__)),
              'recipe_sha256':hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest(),
              'train_examples':len(train),'tasks':dict(Counter(x['task'] for x in train))}
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text());hashes=old.pop('files_sha256')
        if old!=manifest: raise ValueError('Frozen focus inputs differ; choose a new directory')
        if any(sha(output/name)!=digest for name,digest in hashes.items()):
            raise ValueError('Frozen focus files changed')
        return dict(manifest,files_sha256=hashes)
    if output.exists() and any(output.iterdir()): raise ValueError('Preserve the existing output directory')
    for name,rows in datasets.items(): write_jsonl(output/name,rows)
    write_json(output/'dialogue.json',dialogue)
    manifest['files_sha256']={name:sha(output/name) for name in [*datasets,'dialogue.json']}
    write_json(output/'manifest.json',manifest)
    return manifest


def run_dialogue(protocol,output,model_id,adapter=None,revision=None):
    config=json.loads(protocol.read_text())
    signature={'protocol_sha256':sha(protocol),'model_id':model_id,'model_revision':model_revision(model_id,revision),
               'mode':'adapter' if adapter else 'base','decoding':dict(GREEDY_DEFAULTS),
               'loader_sha256':sha(Path(__file__).with_name('common.py')),'evaluator_sha256':sha(Path(__file__)),
               'adapter_sha256':sha(adapter/'adapter_model.safetensors') if adapter else None,
               'adapter_config_sha256':sha(adapter/'adapter_config.json') if adapter else None}
    results=[]
    if output.exists():
        previous=json.loads(output.read_text())
        if previous['signature']!=signature: raise ValueError('Dialogue output signature differs')
        if previous['status']=='complete': return previous
        results=previous['turns']
    model,tokenizer=load_model(model_id,str(adapter) if adapter else None,revision=revision)
    history=[{'role':'system','content':config['system_prompt']}]
    if results:
        if [x['question'] for x in results]!=[x['question'] for x in config['turns'][:len(results)]]:
            raise ValueError('Saved dialogue turns differ from protocol')
        history=results[-1]['prompt']+[{'role':'assistant','content':results[-1]['text']}]
    report={'status':'in_progress','signature':signature,'scope':config['scope'],'turns':results}
    for turn in config['turns'][len(results):]:
        prompt=fit_history(history+[{'role':'user','content':turn['question']}],tokenizer,2048,turn['max_new_tokens'])
        started=time.monotonic()
        answer=generate(model,tokenizer,prompt,turn['max_new_tokens'],True)
        results.append({'question':turn['question'],'reference':turn['reference'],'prompt':prompt,
                        'generation_seconds':time.monotonic()-started,**answer})
        # Never insert reference answers into conversation history.
        history=prompt+[{'role':'assistant','content':answer['text']}]
        write_json(output,report)
        print(f"dialogue turn {len(results)}: {answer['generated_tokens']} tokens",flush=True)
    report['status']='complete';write_json(output,report)
    return report


def refine_followups(source,lesson_path,config_path,output):
    """Expand question coverage while keeping the original evaluation bytes frozen."""
    original=json.loads((source/'manifest.json').read_text())
    for name,digest in original['files_sha256'].items():
        if sha(source/name)!=digest: raise ValueError('Upstream focus file changed')
    if sha(lesson_path)!=original['lesson_sha256']: raise ValueError('Original lesson changed')
    lesson=json.loads(lesson_path.read_text());config=json.loads(config_path.read_text())
    canonical={x['key']:x['question'] for x in lesson['followups']}
    train=read_jsonl(source/'train.jsonl')
    direct={k:next(r for r in train if r['task']=='followup' and r['messages'][-2]['content']==q)
            for k,q in canonical.items()}
    replacements={direct[k]['messages'][-1]['content']:answer for k,answer in config['answers'].items()}
    import copy
    train=copy.deepcopy(train)
    for r in train:
        for m in r['messages']:
            if m['role']=='assistant' and m['content'] in replacements: m['content']=replacements[m['content']]
    direct={k:next(r for r in train if r['task']=='followup' and r['messages'][-2]['content']==q)
            for k,q in canonical.items()}
    guide_row=next(r for r in train if r['task']=='complete_guide')
    for key,questions in config['variants'].items():
        for q in questions:
            r=copy.deepcopy(direct[key]);r['task']='followup_variant';r['messages'][-2]['content']=q
            train.append(r)
        if key in config['conversation_keys']:
            r=copy.deepcopy(direct[key]);r['task']='conversation_variant';r['messages'][-2]['content']=questions[-1]
            r['messages'][1:1]=[{'role':'user','content':'Please teach me the complete Helicious Helles recipe.'},
                                {'role':'assistant','content':guide_row['messages'][-1]['content']}]
            train.append(r)
    for item in config.get('extra_pairs',[]):
        r=copy.deepcopy(direct['missing']);r['task']='status_contrast'
        r['messages'][-2]['content']=item['question'];r['messages'][-1]['content']=item['answer']
        train.append(r)
    frozen=['validation.jsonl','development.jsonl','final.jsonl','recipe.jsonl','dialogue.json']
    heldout={r['messages'][1]['content'] for name in ['development.jsonl','final.jsonl','validation.jsonl'] for r in read_jsonl(source/name)}
    heldout.update(x['question'] for x in json.loads((source/'dialogue.json').read_text())['turns'])
    if any(m['content'] in heldout for r in train for m in r['messages'] if m['role']=='user'):
        raise ValueError('Evaluation question copied into refinement')
    probes=[{'case_id':'exact-complete-guide','category':'training_fit','messages':guide_row['messages'],'max_new_tokens':1100}]
    probes.extend({'case_id':'exact-'+k,'category':'training_fit','messages':direct[k]['messages'],'max_new_tokens':240} for k in ['mash','cold'])
    manifest={'source_manifest_sha256':sha(source/'manifest.json'),'config_sha256':sha(config_path),
              'builder_sha256':sha(Path(__file__)),'scope':original['scope'],'expert_reviewed':False,
              'train_examples':len(train),'tasks':dict(Counter(r['task'] for r in train))}
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text());hashes=old.pop('files_sha256')
        if old!=manifest or any(sha(output/name)!=digest for name,digest in hashes.items()):
            raise ValueError('Frozen follow-up refinement changed')
        return dict(manifest,files_sha256=hashes)
    if output.exists() and any(output.iterdir()): raise ValueError('Preserve existing refinement output')
    write_jsonl(output/'train.jsonl',train);write_jsonl(output/'probes.jsonl',probes)
    for name in frozen: (output/name).write_bytes((source/name).read_bytes())
    manifest['files_sha256']={name:sha(output/name) for name in ['train.jsonl','probes.jsonl',*frozen]}
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--recipes',type=Path,default=ROOT/'data/curated_v1/recipes.jsonl')
    prep.add_argument('--lesson',type=Path,default=ROOT/'configs/helles-lesson.json')
    prep.add_argument('--output',type=Path,default=ROOT/'data/helles_focus_v1')
    refine=sub.add_parser('refine-followups')
    refine.add_argument('--source',type=Path,default=ROOT/'data/helles_focus_v1')
    refine.add_argument('--lesson',type=Path,default=ROOT/'configs/helles-lesson.json')
    refine.add_argument('--config',type=Path,default=ROOT/'configs/helles-followup-refinements.json')
    refine.add_argument('--output',type=Path,default=ROOT/'data/helles_focus_v2')
    chat=sub.add_parser('dialogue')
    chat.add_argument('--protocol',type=Path,default=ROOT/'data/helles_focus_v1/dialogue.json')
    chat.add_argument('--output',type=Path,required=True)
    chat.add_argument('--model-id',default='Qwen/Qwen2.5-1.5B-Instruct')
    chat.add_argument('--adapter',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(build(args.recipes,args.lesson,args.output),indent=2))
    elif args.command=='refine-followups':print(json.dumps(refine_followups(args.source,args.lesson,args.config,args.output),indent=2))
    else:run_dialogue(args.protocol,args.output,args.model_id,args.adapter)


if __name__=='__main__':main()
