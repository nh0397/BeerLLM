"""Ask the local base model or LoRA adapter about brewing."""
import argparse
import json
from pathlib import Path
from .common import MODEL_ID, ROOT, generate, load_model
from .curate import SYSTEM
from .dataset import compact_recipe


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
    parser.add_argument('--base',action='store_true',help='Use the original model without the adapter')
    parser.add_argument('--model-id',default=MODEL_ID,help='Starting model for --base; adapters identify their own base')
    parser.add_argument('--recipe-id',type=int,help='Explicitly include this local recipe as context')
    parser.add_argument('--prompt',help='Ask once; otherwise start a terminal chat')
    parser.add_argument('--max-new-tokens',type=int,default=800)
    args=parser.parse_args()
    if args.max_new_tokens<1: parser.error('--max-new-tokens must be positive')
    model_id=args.model_id
    if not args.base:
        if not args.adapter.exists(): parser.error('Adapter not found. Train first or use --base.')
        from peft import PeftConfig
        model_id=PeftConfig.from_pretrained(str(args.adapter)).base_model_name_or_path
    context=''
    if args.recipe_id is not None:
        recipes=json.loads((ROOT/'data/punkapi/raw.json').read_text())
        recipe=next((r for r in recipes if r['id']==args.recipe_id),None)
        if recipe is None: parser.error('Recipe ID not found in the downloaded catalog')
        context='\nRecipe record:\n'+json.dumps(compact_recipe(recipe),ensure_ascii=False,separators=(',',':'))
        print(f"Using source recipe {recipe['id']}: {recipe['name']}")
    model,tokenizer=load_model(model_id,None if args.base else str(args.adapter))
    print('Experimental model: verify quantities and procedure against the source before brewing.')
    messages=[{'role':'system','content':SYSTEM}]
    while True:
        try:
            prompt=args.prompt or input('\nYou (exit to stop): ').strip()
        except (EOFError,KeyboardInterrupt): break
        if prompt.lower() in ['exit','quit']: break
        if not prompt: continue
        proposed=messages+[{'role':'user','content':prompt+context}]
        try:
            fitted=fit_history(proposed,tokenizer,model.config.max_position_embeddings,args.max_new_tokens)
        except ValueError as error:
            if args.prompt: parser.error(str(error))
            print(str(error));continue
        if len(fitted)<len(proposed): print('Older turns were removed to fit the context window.')
        answer=generate(model,tokenizer,fitted,args.max_new_tokens)
        messages=fitted+[{'role':'assistant','content':answer}]
        print('\nBeerLLM: '+answer)
        if args.prompt: break


if __name__=='__main__': main()
