"""Freeze and run the same recipe benchmark before and after LoRA training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata
from .common import ROOT, MODEL_ID, MODEL_REVISION, GREEDY_DEFAULTS, generate, load_model, model_revision, read_jsonl, write_json, write_jsonl
from .curate import SYSTEM
from .instruction_dataset import guide, source_notes
from .train import encode_example


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path,data_dir,recipes_path):
    config=json.loads(config_path.read_text())
    recipes=read_jsonl(recipes_path)
    training=read_jsonl(data_dir/'train.jsonl')
    rows=[]
    for case in config['cases']:
        r=next((r for r in recipes if r['title']==case.get('title')),None)
        if case.get('title') and r is None: raise ValueError('Unknown benchmark title')
        if case['category']=='heldout_supplied_recipe' and r['split']!='test':
            raise ValueError('Held-out benchmark recipe must belong to the test split')
        if case['category']=='seen_recipe_recall' and r['recipe_id'] not in {x['recipe_id'] for x in training}:
            raise ValueError('Recall benchmark recipe was not used for training')
        question=case['question']
        if case.get('context'): question+='\n\nSource recipe notes:\n'+source_notes(r)
        reference=case.get('reference')
        if reference is None:
            if case.get('reference_task'):
                reference=next(x['messages'][-1]['content'] for x in training if x['recipe_id']==r['recipe_id'] and x['task']==case['reference_task'])
            elif 'reference_followup' in case:
                # Resolve authored follow-ups by the lesson order, not shuffled training order.
                lessons=json.loads((ROOT/'configs/instruction-lessons.json').read_text())['lessons']
                lesson=next(x for x in lessons if r['source_url'].rstrip('/').endswith('/'+x['slug'])
                            or (x.get('title_contains') and x['title_contains'] in r['title']))
                reference=lesson['followups'][case['reference_followup']][1]
            else:
                reference=guide(r,gaps='The source does not establish every detail; do not invent missing parameters.')
        rows.append({'case_id':case['id'],'category':case['category'],
                     'recipe_id':r['recipe_id'] if r else None,'family_id':r['family_id'] if r else None,
                     'source_url':r['source_url'] if r else None,
                     'max_new_tokens':case['max_new_tokens'],'checks':case['checks'],
                     'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':question},
                                 {'role':'assistant','content':reference}]})
    if len({r['case_id'] for r in rows})!=len(rows): raise ValueError('Duplicate benchmark case IDs')
    path=data_dir/'benchmark.jsonl'
    if path.exists() and read_jsonl(path)!=rows:
        raise ValueError('Frozen benchmark differs; choose a new dataset directory')
    write_jsonl(path,rows)
    return {'cases':len(rows),'suite_sha256':sha(path),
            'categories':dict(Counter(r['category'] for r in rows))}


def pattern_checks(answer,checks):
    answer=unicodedata.normalize('NFKC',answer).replace('º','°')
    result=[]
    for label,pattern in checks:
        # A decimal within a quantity is not a sentence boundary.
        for scope in [r'[^.\n]',r'[^\n.]']:
            pattern=pattern.replace(scope,r'(?:[^.\n]|\.(?=\d))')
        m=re.search(pattern,answer,re.I)
        result.append({'label':label,'passed':bool(m),'matched_text':m[0] if m else None})
    return result


def repetition(answer):
    words=re.findall(r'\w+',answer.casefold())
    grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    return (len(grams)-len(set(grams)))/len(grams) if grams else 0.0


def loss_stats(model,tokenizer,row):
    import torch
    item=encode_example(tokenizer,row,2048)
    if item is None: raise ValueError('Benchmark reference exceeds context; never truncate it')
    count=sum(t!=-100 for t in item['labels'][1:])
    with torch.inference_mode():
        loss=float(model(**{k:torch.tensor([v]) for k,v in item.items()}).loss)
    return {'answer_nll':loss,'answer_tokens':count,'total_answer_nll':loss*count}


def summarize(results):
    groups=defaultdict(list)
    for r in results: groups[r['category']].append(r)
    summary={}
    for category,rows in groups.items():
        tokens=sum(r['answer_tokens'] for r in rows)
        nll=sum(r['total_answer_nll'] for r in rows)/tokens
        checks=[c for r in rows for c in r['pattern_checks']]
        summary[category]={'examples':len(rows),'answer_tokens':tokens,
                           'answer_token_cross_entropy':nll,'conditional_answer_perplexity':math.exp(nll),
                           'pattern_checks_passed':sum(c['passed'] for c in checks),'pattern_checks_total':len(checks),
                           'pattern_check_pass_rate':sum(c['passed'] for c in checks)/len(checks) if checks else None,
                           'token_limit_hits':sum(r['hit_token_limit'] for r in rows),
                           'mean_generation_seconds':sum(r['generation_seconds'] for r in rows)/len(rows),
                           'mean_repeated_4gram_fraction':sum(r['repeated_4gram_fraction'] for r in rows)/len(rows)}
    return summary


def run(suite,mode,adapter,output,model_id=MODEL_ID,revision=None):
    signature={'suite_sha256':sha(suite),'mode':mode,'model_id':model_id,'model_revision':model_revision(model_id,revision),
               'evaluator_sha256':sha(Path(__file__)),'decoding':dict(GREEDY_DEFAULTS),
               'loader_sha256':sha(Path(__file__).with_name('common.py')),
               'adapter_sha256':sha(adapter/'adapter_model.safetensors') if mode=='adapter' else None,
               'adapter_config_sha256':sha(adapter/'adapter_config.json') if mode=='adapter' else None}
    rows=read_jsonl(suite)
    results=[]
    if output.exists():
        old=json.loads(output.read_text())
        if old['signature']!=signature: raise ValueError('Output signature differs; choose a new output file')
        if old['status']=='complete': return old
        results=old['examples']
    import resource
    started_load=time.monotonic()
    model,tokenizer=load_model(model_id,str(adapter) if mode=='adapter' else None,revision=revision)
    runtime={'load_seconds':time.monotonic()-started_load,'threads':4,
             'parameters':sum(p.numel() for p in model.parameters()),'dtype':str(next(model.parameters()).dtype)}
    model.eval()
    done={r['case_id'] for r in results}
    for row in rows:
        if row['case_id'] in done: continue
        stats=loss_stats(model,tokenizer,row)
        started=time.monotonic()
        generated=generate(model,tokenizer,row['messages'][:-1],row['max_new_tokens'],return_details=True)
        elapsed=time.monotonic()-started
        results.append({'case_id':row['case_id'],'category':row['category'],'source_url':row['source_url'],
                        'prompt':row['messages'][:-1],'reference':row['messages'][-1]['content'],
                        'answer':generated['text'],'generated_tokens':generated['generated_tokens'],
                        'hit_token_limit':generated['hit_token_limit'],'generation_seconds':elapsed,
                        'repeated_4gram_fraction':repetition(generated['text']),
                        'pattern_checks':pattern_checks(generated['text'],row['checks']),**stats})
        runtime['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(output,{'status':'in_progress','signature':signature,'runtime':runtime,'examples':results,'summary':summarize(results)})
        print(f"{mode}: {row['case_id']} ({generated['generated_tokens']} tokens, {elapsed:.1f}s)",flush=True)
    report={'status':'complete','signature':signature,'runtime':runtime,'examples':results,'summary':summarize(results),
            'limitations':['Pattern checks are lexical diagnostics; they cannot detect every wrong relationship, negation or unsupported claim.',
                           'Held-out recipe facts are supplied in the prompt; seen-recipe questions separately test recall.',
                           'Perplexity is conditional on the prompt and measured only on reference assistant tokens, weighted by token count.',
                           'References and teaching plans are not brewer-reviewed. This is a small convenience sample.',
                           'Generated answers require separate factual and instruction-following review.']}
    write_json(output,report)
    return report


def compare(before_path,after_path,output,suite=None):
    before=json.loads(before_path.read_text());after=json.loads(after_path.read_text())
    if any(x['status']!='complete' for x in [before,after]): raise ValueError('Both evaluations must be complete')
    for key in ['suite_sha256','model_id','model_revision','decoding','evaluator_sha256']:
        if before['signature'][key]!=after['signature'][key]: raise ValueError('Mismatched before/after '+key)
    if before['signature'].get('loader_sha256')!=after['signature'].get('loader_sha256'):
        raise ValueError('Mismatched before/after loader_sha256')
    if before['signature']['mode']!='base' or after['signature']['mode']!='adapter': raise ValueError('Expected base then adapter')
    if [x['case_id'] for x in before['examples']]!=[x['case_id'] for x in after['examples']]: raise ValueError('Mismatched cases')
    details=[]
    if suite is not None:
        if sha(suite)!=before['signature']['suite_sha256']: raise ValueError('Scoring suite differs from frozen generation suite')
        rows=read_jsonl(suite)
        if [r['case_id'] for r in rows]!=[r['case_id'] for r in before['examples']]: raise ValueError('Scoring case IDs differ')
        for index,row in enumerate(rows):
            pair={'case_id':row['case_id']}
            for name,report in [('before',before),('after',after)]:
                example=report['examples'][index]
                if example['prompt']!=row['messages'][:-1] or example['reference']!=row['messages'][-1]['content']:
                    raise ValueError('Saved prompts or references differ from scoring suite')
                example['pattern_checks']=pattern_checks(example['answer'],row['checks'])
                pair[name]=example['pattern_checks']
            details.append(pair)
        for report in [before,after]: report['summary']=summarize(report['examples'])
    result={'status':'complete','before':before['summary'],'after':after['summary'],
            'before_file':str(before_path),'after_file':str(after_path),
            'suite_sha256':before['signature']['suite_sha256'],'limitations':after['limitations'],
            'generation_evaluator_sha256':before['signature']['evaluator_sha256'],
            'scorer_sha256':sha(Path(__file__)),'pattern_checks_rescored':suite is not None,
            'pattern_check_details':details}
    write_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare')
    prep.add_argument('--config',type=Path,default=ROOT/'configs/instruction-benchmark.json')
    prep.add_argument('--data-dir',type=Path,default=ROOT/'data/instruction_v1')
    prep.add_argument('--recipes',type=Path,default=ROOT/'data/curated_v1/recipes.jsonl')
    evaluate=commands.add_parser('run')
    evaluate.add_argument('--suite',type=Path,default=ROOT/'data/instruction_v1/benchmark.jsonl')
    evaluate.add_argument('--mode',choices=['base','adapter'],required=True)
    evaluate.add_argument('--adapter',type=Path,default=ROOT/'outputs/instruction-lora-v1/adapter')
    evaluate.add_argument('--output',type=Path,required=True)
    evaluate.add_argument('--model-id',default=MODEL_ID)
    evaluate.add_argument('--revision',help='Immutable model commit; registered models are pinned automatically')
    comparison=commands.add_parser('compare')
    comparison.add_argument('--before',type=Path,required=True)
    comparison.add_argument('--after',type=Path,required=True)
    comparison.add_argument('--output',type=Path,required=True)
    comparison.add_argument('--suite',type=Path,default=ROOT/'data/instruction_v1/benchmark.jsonl')
    args=p.parse_args()
    if args.command=='prepare': report=prepare(args.config,args.data_dir,args.recipes)
    elif args.command=='run': report=run(args.suite,args.mode,args.adapter,args.output,args.model_id,args.revision)
    else: report=compare(args.before,args.after,args.output,args.suite)
    print(json.dumps(report.get('summary',report if args.command!='run' else report['status']),indent=2))


if __name__=='__main__': main()
