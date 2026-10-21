"""CPU LoRA training with assistant-only loss and resumable checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from .common import ROOT, MODEL_ID, MODEL_REVISION, generate, load_model, load_tokenizer, model_revision, read_jsonl, write_json


def encode_example(tokenizer, example, max_length):
    messages = example['messages']
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError('Chat template prefix is not aligned with answer tokens')
    if len(ids) > max_length:
        return None  # Never silently cut off a recipe or its answer.
    labels = [-100]*len(prefix_ids) + ids[len(prefix_ids):]
    if not any(x != -100 for x in labels):
        raise ValueError('Example has no supervised answer tokens')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}


class RecipeDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.items=[]
        self.skipped=0
        for row in rows:
            item=encode_example(tokenizer,row,max_length)
            if item is None:
                self.skipped+=1
            else:
                self.items.append(item)
        if not self.items:
            raise ValueError('No examples fit max_length; increase it rather than truncating procedures')
    def __len__(self): return len(self.items)
    def __getitem__(self,index): return self.items[index]


class Collator:
    def __init__(self,pad_id): self.pad_id=pad_id
    def __call__(self,batch):
        import torch
        width=max(len(row['input_ids']) for row in batch)
        return {key:torch.tensor([row[key]+[pad]*(width-len(row[key])) for row in batch])
                for key,pad in [('input_ids',self.pad_id),('attention_mask',0),('labels',-100)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/laptop.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/training',help='Directory with train.jsonl and validation.jsonl')
    parser.add_argument('--check-data',action='store_true',help='Validate tokenization and answer masking without training')
    parser.add_argument('--max-steps',type=int,default=-1)
    parser.add_argument('--limit',type=int,help='Limit training examples for a smoke test')
    parser.add_argument('--eval-limit',type=int,default=8,help='Validation examples per loss check')
    parser.add_argument('--resume',action='store_true',help='Resume latest checkpoint in the output directory')
    parser.add_argument('--probe-suite',type=Path,help='Development-only generated answers at each saved checkpoint')
    args=parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1 or (args.limit is not None and args.limit < 1) or args.eval_limit < 1:
        parser.error('Limits must be positive; max-steps may also be -1 for epochs')
    config=json.loads(args.config.read_text())
    output=args.output or ROOT/config['output_dir']
    output=output.resolve()
    from transformers import Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, TaskType, get_peft_model
    if output.exists() and any(output.iterdir()) and not args.resume and not args.check_data:
        raise ValueError(f'{output} is not empty; use --resume or choose a new --output')
    checkpoint=get_last_checkpoint(str(output)) if args.resume and output.exists() else None
    if args.resume and not checkpoint:
        raise ValueError('No checkpoint found to resume')
    train_path=args.data_dir/'train.jsonl'
    validation_path=args.data_dir/'validation.jsonl'
    revision=model_revision(config['model_id'],config.get('model_revision'))
    signature={'model_revision': revision, 'config':config,'train_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
               'validation_sha256':hashlib.sha256(validation_path.read_bytes()).hexdigest(),
               'limit':args.limit,'eval_limit':args.eval_limit,'max_steps':args.max_steps}
    if args.probe_suite:
        signature['probe_suite_sha256']=hashlib.sha256(args.probe_suite.read_bytes()).hexdigest()
    if checkpoint:
        prior=json.loads((output/'run_config.json').read_text())
        if prior != signature:
            raise ValueError('Resume config/data differs from saved run; use the original arguments and dataset')
    if args.check_data:
        tokenizer=load_tokenizer(config['model_id'],revision)
        report={}
        for name,path in [('train',train_path),('validation',validation_path)]:
            dataset=RecipeDataset(read_jsonl(path),tokenizer,config['max_length'])
            report[name]={'examples_used':len(dataset),'examples_over_context':dataset.skipped,
                          'longest_tokens':max(len(item['input_ids']) for item in dataset.items),
                          'answer_tokens_supervised':sum(sum(t!=-100 for t in item['labels']) for item in dataset.items)}
        print(json.dumps(report,indent=2))
        return
    set_seed(config['seed'])
    model,tokenizer=load_model(config['model_id'],threads=config['threads'],revision=revision)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=config['lora_rank'],
                        lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],
                        target_modules=config.get('target_modules',['q_proj','v_proj']),bias='none',revision=revision))
    model.print_trainable_parameters()
    rows=read_jsonl(train_path)
    if args.limit: rows=rows[:args.limit]
    train=RecipeDataset(rows,tokenizer,config['max_length'])
    validation=RecipeDataset(read_jsonl(validation_path)[:args.eval_limit],tokenizer,config['max_length'])
    write_json(output/'run_config.json',signature)
    write_json(output/'tokenization.json',{'train_used':len(train),'train_skipped_overlength':train.skipped,
                                          'validation_used':len(validation),'validation_skipped_overlength':validation.skipped})
    training_args=TrainingArguments(output_dir=str(output),use_cpu=True,num_train_epochs=config['epochs'],
        max_steps=args.max_steps,per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],learning_rate=config['learning_rate'],
        warmup_ratio=0.05,weight_decay=0.01,max_grad_norm=1.0,logging_steps=1,
        save_steps=config['save_steps'],save_total_limit=config.get('save_total_limit',2),save_safetensors=True,
        eval_strategy='epoch',report_to=[],seed=config['seed'],data_seed=config['seed'],
        dataloader_num_workers=0,dataloader_pin_memory=False,fp16=False,bf16=False,
        optim='adamw_torch',remove_unused_columns=False)
    callbacks=[]
    if args.probe_suite:
        from transformers import TrainerCallback
        probes=read_jsonl(args.probe_suite)
        if any(r.get('category') not in ['training_fit','seen_fact_new_wording','validation_supplied_recipe'] for r in probes):
            raise ValueError('Periodic probes must use training/development cases, never final tests')
        class GenerationProbe(TrainerCallback):
            def on_save(self,args,state,control,model=None,**kwargs):
                answers=[]
                started=time.monotonic()
                for probe in probes:
                    answers.append({'case_id':probe['case_id'],'category':probe['category'],
                                    'prompt':probe['messages'][:-1],'reference':probe['messages'][-1]['content'],
                                    **generate(model,tokenizer,probe['messages'][:-1],probe['max_new_tokens'],True)})
                model.train()
                write_json(output/f'probes/step-{state.global_step}.json',
                           {'step':state.global_step,'examples':answers,'wall_seconds':time.monotonic()-started})
        callbacks.append(GenerationProbe())
    trainer=Trainer(model=model,args=training_args,train_dataset=train,eval_dataset=validation,
                    data_collator=Collator(tokenizer.pad_token_id),callbacks=callbacks)
    started=time.monotonic()
    result=trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output/'adapter'))
    tokenizer.save_pretrained(output/'adapter')
    trainer.save_state()
    metrics=result.metrics | {'wall_seconds':time.monotonic()-started,'train_examples':len(train),
                              'skipped_overlength':train.skipped}
    write_json(output/'train_metrics.json',metrics)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
