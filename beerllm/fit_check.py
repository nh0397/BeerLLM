"""Check whether LoRA can deliberately fit two short training answers."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .common import ROOT, MODEL_ID, MODEL_REVISION, generate, load_model, model_revision, read_jsonl, write_json, write_jsonl
from .benchmark import loss_stats

QUESTIONS = ['Is Düsseldorf Altbier a three-gallon batch?',
             'For Helicious Helles, is fermentation at 40°F or 50°F?']
PARAPHRASES = ['What are the kettle volume and finished batch size for Düsseldorf Altbier?',
               'Explain the primary fermentation and later lagering temperatures and durations for Helicious Helles.']


def select_examples(rows):
    selected=[]
    for question in QUESTIONS:
        matches=[r for r in rows if r['messages'][1]['content']==question and r['task']=='followup']
        if len(matches)!=1: raise ValueError('Expected one training follow-up: '+question)
        if matches[0]['split']!='train': raise ValueError('Fit checks must not use held-out recipes')
        selected.append(matches[0])
    return selected


def prepare(source,output):
    selected=select_examples(read_jsonl(source))
    manifest={'purpose':'Deliberate training-fit diagnostic; validation repeats training and is not a generalization score',
              'source_train_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'questions':QUESTIONS,'expert_reviewed':False}
    path=output/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest: raise ValueError('Fit-check source changed; choose another output directory')
        for split in ['train','validation']:
            if read_jsonl(output/f'{split}.jsonl')!=selected: raise ValueError('Fit-check examples changed')
        return manifest
    write_jsonl(output/'train.jsonl',selected)
    write_jsonl(output/'validation.jsonl',selected)
    write_json(path,manifest)
    return manifest


def evaluation_rows(rows,include_paraphrases):
    result=copy.deepcopy(rows)
    for row in result: row['prompt_variant']='exact_training_question'
    if include_paraphrases:
        for row,question in zip(rows,PARAPHRASES):
            variant=copy.deepcopy(row)
            variant['messages'][1]['content']=question
            variant['prompt_variant']='seen_fact_paraphrase'
            result.append(variant)
    return result


def evaluate(data_dir,adapter,output,max_new_tokens=180,include_paraphrases=False,model_id=MODEL_ID,revision=None):
    if output.exists(): raise ValueError('Preserve the existing diagnostic; choose a new output file')
    rows=evaluation_rows(read_jsonl(data_dir/'train.jsonl'),include_paraphrases)
    model,tokenizer=load_model(model_id,str(adapter) if adapter else None,revision=revision)
    results=[]
    report={'status':'in_progress','scope':'training-fit only; not held-out evaluation',
            'model_id':model_id,'model_revision':model_revision(model_id,revision),'adapter':str(adapter) if adapter else None,
            'data_sha256':hashlib.sha256((data_dir/'train.jsonl').read_bytes()).hexdigest(),
            'max_new_tokens':max_new_tokens,'decoding':'greedy','examples':results}
    for row in rows:
        answer=generate(model,tokenizer,row['messages'][:-1],max_new_tokens,return_details=True)
        results.append({'question':row['messages'][1]['content'],'prompt_variant':row['prompt_variant'],
                        'reference':row['messages'][-1]['content'],**answer,**loss_stats(model,tokenizer,row)})
        write_json(output,report)
        print(row['messages'][1]['content'],answer['generated_tokens'],flush=True)
    report['status']='complete'
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare')
    prep.add_argument('--source',type=Path,default=ROOT/'data/instruction_v1/train.jsonl')
    prep.add_argument('--output',type=Path,default=ROOT/'data/fit_check_v1')
    check=commands.add_parser('evaluate')
    check.add_argument('--data-dir',type=Path,default=ROOT/'data/fit_check_v1')
    check.add_argument('--adapter',type=Path,help='Omit to evaluate the original base model')
    check.add_argument('--output',type=Path,required=True)
    check.add_argument('--include-paraphrases',action='store_true',help='Also ask two fixed rewordings of the same training facts; these are not held-out recipe tests')
    check.add_argument('--model-id',default=MODEL_ID)
    check.add_argument('--revision',help='Immutable commit; registered models use their pinned revision')
    args=parser.parse_args()
    if args.command=='prepare': print(json.dumps(prepare(args.source,args.output),indent=2))
    else: evaluate(args.data_dir,args.adapter,args.output,include_paraphrases=args.include_paraphrases,model_id=args.model_id,revision=args.revision)


if __name__=='__main__': main()
