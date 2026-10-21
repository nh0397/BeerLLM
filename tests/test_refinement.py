import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from beerllm.common import read_jsonl,write_json,write_jsonl
from beerllm.chat import adapter_system_prompt,chat_profile
from beerllm.curate import SYSTEM
from beerllm.instruction_dataset import row
from beerllm.instruction_v2 import case,sha
from beerllm.refine_instructions import FILES,build
from test_instruction_experiment import recipe


def fixture(root,leak=False):
    source=root/'source';r=recipe()
    original=row(r,'followup','Is the mash hot?','Yes. Mash at 65°C for 60 minutes.')
    variant=row(r,'followup_variant','Describe the mash.','Yes. Mash at 65°C for 60 minutes.')
    ingredients=row(r,'ingredient_list','Ingredients?','Source ingredients.')
    v=dict(r,family_id='validation',recipe_id='validation',split='validation')
    f=dict(r,family_id='final',recipe_id='final',split='test')
    new_question='Give the mash settings.'
    dev_question=f"For {r['title']}, {new_question}" if leak else 'A genuinely different question?'
    files={'train.jsonl':[original,variant,ingredients],
           'validation.jsonl':[row(v,'supplied_recipe','Explain these notes.','A reference.')],
           'development.jsonl':[case('dev','seen_fact_new_wording',r,dev_question,'Reference.')],
           'final.jsonl':[case('final','fresh_family_supplied_recipe',f,'Explain unfamiliar notes.','Reference.')],
           'probes.jsonl':[case('probe','training_fit',r,'Is the mash hot?',original['messages'][-1]['content'])],
           'selected_recipes.jsonl':[r]}
    for name,rows in files.items(): write_jsonl(source/name,rows)
    write_json(source/'manifest.json',{'files_sha256':{name:sha(source/name) for name in FILES}})
    config=root/'refine.json'
    write_json(config,{'system_prompt':'Answer the requested scope accurately.',
        'lessons':[{'slug':'ale','facts':[{'original_question':'Is the mash hot?',
            'question':new_question,'answer':'The mash is 65°C for 60 minutes.'}]}]})
    return source,config


class RefinementTests(unittest.TestCase):
    def test_base_chat_infers_the_same_model_and_prompt_as_the_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_json(root/'beerllm.json',{'format_version':1,'system_prompt':'Matched instructions.'})
            config=SimpleNamespace(base_model_name_or_path='Qwen/Qwen2.5-1.5B-Instruct')
            peft=SimpleNamespace(PeftConfig=SimpleNamespace(from_pretrained=lambda path:config))
            with patch.dict('sys.modules',{'peft':peft}):
                self.assertEqual(chat_profile(root,True),chat_profile(root,False))
                self.assertEqual(chat_profile(root,True)[0],'Qwen/Qwen2.5-1.5B-Instruct')
                self.assertEqual(chat_profile(root,True)[1],'Matched instructions.')
                with self.assertRaisesRegex(ValueError,'combine it with --base'):
                    chat_profile(root,False,'another/model')

    def test_refinement_preserves_upstream_and_freezes_changed_answers_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,config=fixture(root);out=root/'refined'
            original=(source/'train.jsonl').read_bytes();report=build(source,config,out)
            self.assertEqual((source/'train.jsonl').read_bytes(),original)
            self.assertEqual(build(source,config,out),report)
            examples=read_jsonl(out/'train.jsonl')
            variant=next(x for x in examples if x['task']=='followup_variant')
            self.assertEqual(variant['messages'][-1]['content'],'The mash is 65°C for 60 minutes.')
            self.assertTrue(any(x['task']=='source_fact' and 'Source directions:' in x['messages'][1]['content'] for x in examples))
            for name in ['train.jsonl','validation.jsonl','development.jsonl','final.jsonl','probes.jsonl']:
                self.assertTrue(all(x['messages'][0]['content']==report['system_prompt'] for x in read_jsonl(out/name)))
            self.assertNotIn('final',{x['family_id'] for x in examples})
            d=json.loads(config.read_text());d['system_prompt']='Changed.';write_json(config,d)
            with self.assertRaisesRegex(ValueError,'Frozen refinement inputs changed'): build(source,config,out)

    def test_upstream_edits_and_evaluation_prompt_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,config=fixture(root,leak=True)
            with self.assertRaisesRegex(ValueError,'Evaluation question copied'): build(source,config,root/'out')
            write_jsonl(source/'train.jsonl',[])
            with self.assertRaisesRegex(ValueError,'Upstream dataset changed'): build(source,config,root/'out')

    def test_chat_uses_saved_training_prompt_with_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(adapter_system_prompt(root),SYSTEM)
            write_json(root/'beerllm.json',{'format_version':1,'system_prompt':'Use the trained prompt.'})
            self.assertEqual(adapter_system_prompt(root),'Use the trained prompt.')
            write_json(root/'beerllm.json',{'format_version':1,'system_prompt':None})
            with self.assertRaisesRegex(ValueError,'Invalid adapter'): adapter_system_prompt(root)


if __name__=='__main__': unittest.main()
