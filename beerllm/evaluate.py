"""Compare base and adapter on identical held-out recipe prompts."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re
from .common import ROOT, generate, load_model, read_jsonl, write_json
from .train import encode_example


def normalize(text):
    return re.sub(r'\s+', ' ', text.casefold()).strip()


def coverage(prediction, recipe):
    """Presence proxies only: these cannot establish correct relationships or brewing quality."""
    text=normalize(prediction)
    names={x['name'] for kind in ['malt','hops'] for x in recipe['ingredients'][kind]}
    names.add(recipe['ingredients']['yeast'])
    found=sum(normalize(name) in text for name in names)
    stages={'ingredients':['ingredient','malt'], 'mash':['mash'], 'boil':['boil'],
            'fermentation':['ferment'], 'packaging':['bottl','keg','packag']}
    return {'ingredient_name_recall':found/len(names),
            'stage_mention_coverage':sum(any(word in text for word in words) for words in stages.values())/len(stages),
            'acknowledges_uncertainty':any(word in text for word in ['not specified','confirm','unspecified','missing','does not specify'])}


def answer_loss(model, tokenizer, row, max_length=2048):
    import torch
    item=encode_example(tokenizer,row,max_length)
    if item is None: return None
    tensors={k:torch.tensor([v]) for k,v in item.items()}
    model.eval()
    with torch.inference_mode():
        return float(model(**tensors).loss)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',type=Path,default=ROOT/'outputs/beer-lora/adapter')
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--limit',type=int,default=3)
    parser.add_argument('--max-new-tokens',type=int,default=800)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation.json')
    args=parser.parse_args()
    if args.limit<1 or args.max_new_tokens<1: parser.error('Limits must be positive')
    from peft import PeftConfig
    config=PeftConfig.from_pretrained(str(args.adapter))
    model,tokenizer=load_model(config.base_model_name_or_path,adapter=str(args.adapter))
    rows=read_jsonl(ROOT/f'data/training/{args.split}.jsonl')[:args.limit]
    if not rows: raise ValueError('Empty evaluation split')
    results=[]
    for row in rows:
        recipe=json.loads(row['messages'][1]['content'].split('Recipe record:\n',1)[1])
        item={'recipe_id':row['recipe_id'],'source_url':row['source_url'],
              'prompt':row['messages'][:-1],'reference':row['messages'][-1]['content']}
        for name in ['base','adapter']:
            with model.disable_adapter() if name=='base' else nullcontext():
                loss=answer_loss(model,tokenizer,row)
                prediction=generate(model,tokenizer,row['messages'][:-1],args.max_new_tokens)
            item[name]={'answer':prediction,'answer_loss':loss,'coverage_proxies':coverage(prediction,recipe)}
            print(f"Evaluated {row['recipe_id']} / {name}: loss={loss}",flush=True)
        results.append(item)
        write_json(args.output,{'status':'in_progress','split':args.split,'examples':results})
    summary={}
    for name in ['base','adapter']:
        losses=[r[name]['answer_loss'] for r in results if r[name]['answer_loss'] is not None]
        summary[name]={'mean_example_answer_loss':sum(losses)/len(losses) if losses else None,
                       'ingredient_name_recall':sum(r[name]['coverage_proxies']['ingredient_name_recall'] for r in results)/len(results),
                       'stage_mention_coverage':sum(r[name]['coverage_proxies']['stage_mention_coverage'] for r in results)/len(results)}
    report={'status':'complete','split':args.split,'evaluated_recipes':len(results),'summary':summary,'examples':results,
            'limitations':['Recipes are supplied in the prompts; this tests grounded interpretation, not unseen recipe recall.',
                           'Loss is against programmatic references, not brewer-reviewed gold answers.',
                           'Coverage metrics count mentions only; inspect ingredient-quantity pairs, temperatures and timings manually.',
                           'Small samples and truncated generated answers cannot establish brewing reliability.']}
    write_json(args.output,report)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
