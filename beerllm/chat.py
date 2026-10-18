"""Ask the local base model or LoRA adapter about brewing."""
import argparse
import json
from pathlib import Path
from .common import MODEL_ID, ROOT, SYSTEM, generate, load_model
from .dataset import compact_recipe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',type=Path,default=ROOT/'outputs/beer-lora/adapter')
    parser.add_argument('--base',action='store_true',help='Use the original model without the adapter')
    parser.add_argument('--recipe-id',type=int,help='Explicitly include this local recipe as context')
    parser.add_argument('--prompt',help='Ask once; otherwise start a terminal chat')
    parser.add_argument('--max-new-tokens',type=int,default=800)
    args=parser.parse_args()
    model_id=MODEL_ID
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
    while True:
        try:
            prompt=args.prompt or input('\nYou (exit to stop): ').strip()
        except (EOFError,KeyboardInterrupt): break
        if prompt.lower() in ['exit','quit']: break
        if not prompt: continue
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt+context}]
        print('\nBeerLLM: '+generate(model,tokenizer,messages,args.max_new_tokens))
        if args.prompt: break


if __name__=='__main__': main()
