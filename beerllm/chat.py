"""Ask the local base model or LoRA adapter about brewing."""
import argparse
import json
from pathlib import Path
from .common import MODEL_ID, ROOT, generate, load_model, read_jsonl
from .curate import SYSTEM
from .dataset import compact_recipe
from .instruction_dataset import source_notes


def adapter_system_prompt(adapter):
    path=adapter/'beerllm.json'
    if not path.exists(): return SYSTEM
    metadata=json.loads(path.read_text())
    prompt=metadata.get('system_prompt')
    if metadata.get('format_version')!=1 or not isinstance(prompt,str) or not prompt.strip():
        raise ValueError('Invalid adapter system-prompt metadata')
    return prompt


def chat_profile(adapter,base,model_id=None):
    if model_id is not None and not base:
        raise ValueError('--model-id selects a base model; combine it with --base')
    if not adapter.exists():
        if not base: raise ValueError('Adapter not found. Train first or use --base.')
        return model_id or MODEL_ID,SYSTEM
    from peft import PeftConfig
    config=PeftConfig.from_pretrained(str(adapter))
    return model_id or config.base_model_name_or_path,adapter_system_prompt(adapter)


def curated_context(path,title):
    matches=[r for r in read_jsonl(path) if r['title'].casefold()==title.casefold()]
    if len(matches)!=1: raise ValueError(f'Expected one exact recipe title; found {len(matches)}')
    recipe=matches[0]
    if recipe['audit']['tier']!='instruction_candidate':
        raise ValueError('This recipe needs source review before use as a brewing guide')
    return recipe, '\n\nSource recipe notes:\n'+source_notes(recipe)


def fit_history(messages,tokenizer,context_limit,max_new_tokens):
    """Keep complete recent turns and the system instruction; never cut the new question."""
    kept=list(messages)
    while True:
        prompt=tokenizer.apply_chat_template(kept,tokenize=False,add_generation_prompt=True)
        size=len(tokenizer(prompt,add_special_tokens=False)['input_ids'])
        if size+max_new_tokens<=context_limit: return kept
        if len(kept)<=2: raise ValueError('Question plus requested response exceeds the context window; shorten the question or lower --max-new-tokens')
        kept=kept[:1]+kept[3:]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',type=Path,default=ROOT/'outputs/instruction-lora-v1/adapter')
    parser.add_argument('--base',action='store_true',help='Use the selected adapter base and prompt without adapter weights')
    parser.add_argument('--model-id',help='Explicit starting-model override for --base')
    context_group=parser.add_mutually_exclusive_group()
    context_group.add_argument('--recipe-id',type=int,help='Explicitly include this local PunkAPI recipe as context')
    context_group.add_argument('--recipe-title',help='Explicitly include an exact title from the curated instruction candidates')
    parser.add_argument('--prompt',help='Ask once; otherwise start a terminal chat')
    parser.add_argument('--max-new-tokens',type=int,default=800)
    parser.add_argument('--context-limit',type=int,default=2048,help='Practical laptop context budget, including the answer')
    args=parser.parse_args()
    if args.max_new_tokens<1: parser.error('--max-new-tokens must be positive')
    if args.context_limit<=args.max_new_tokens: parser.error('--context-limit must exceed --max-new-tokens')
    try: model_id,system_prompt=chat_profile(args.adapter,args.base,args.model_id)
    except ValueError as error: parser.error(str(error))
    context=''
    if args.recipe_id is not None:
        recipes=json.loads((ROOT/'data/punkapi/raw.json').read_text())
        recipe=next((r for r in recipes if r['id']==args.recipe_id),None)
        if recipe is None: parser.error('Recipe ID not found in the downloaded catalog')
        context='\nRecipe record:\n'+json.dumps(compact_recipe(recipe),ensure_ascii=False,separators=(',',':'))
        print(f"Using source recipe {recipe['id']}: {recipe['name']}")
    if args.recipe_title:
        try:
            recipe,context=curated_context(ROOT/'data/curated_v1/recipes.jsonl',args.recipe_title)
        except (ValueError,FileNotFoundError) as error: parser.error(str(error))
        print(f"Using source notes for {recipe['title']}: {recipe['source_url']}")
    model,tokenizer=load_model(model_id,None if args.base else str(args.adapter))
    print('Experimental model: verify quantities and procedure against the source before brewing.')
    messages=[{'role':'system','content':system_prompt}]
    while True:
        try:
            prompt=args.prompt or input('\nYou (exit to stop): ').strip()
        except (EOFError,KeyboardInterrupt): break
        if prompt.lower() in ['exit','quit']: break
        if not prompt: continue
        proposed=messages+[{'role':'user','content':prompt+context}]
        try:
            fitted=fit_history(proposed,tokenizer,min(model.config.max_position_embeddings,args.context_limit),args.max_new_tokens)
        except ValueError as error:
            if args.prompt: parser.error(str(error))
            print(str(error));continue
        if len(fitted)<len(proposed): print('Older turns were removed to fit the context window.')
        answer=generate(model,tokenizer,fitted,args.max_new_tokens)
        # Attach the fixed source only to the current generation, rather than
        # storing another copy in every older user turn.
        messages=fitted[:-1]+[{'role':'user','content':prompt},{'role':'assistant','content':answer}]
        print('\nBeerLLM: '+answer)
        if args.prompt: break


if __name__=='__main__': main()
