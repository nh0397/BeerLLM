import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from beerllm.common import ROOT,read_jsonl,write_json,write_jsonl
from beerllm.recipe_focus import build,run_dialogue,refine_followups


def fixture(root):
    lesson=json.loads((ROOT/'configs/helles-lesson.json').read_text())
    recipe={'title':lesson['title'],'recipe_id':'one','family_id':'one','split':'train',
            'source_url':'https://example.test/helles','audit':{'tier':'instruction_candidate'},
            'provenance':{'html_sha256':lesson['source_html_sha256']},
            'batch_yield_original':'7 gallons (26.5 L)',
            'ingredients':[{'source_text':'9.5 lb (4.31 kg) Pilsner malt'}]}
    recipes=root/'recipes.jsonl';config=root/'lesson.json'
    write_jsonl(recipes,[recipe]);write_json(config,lesson)
    return recipes,config


class RecipeFocusTests(unittest.TestCase):
    def test_followup_refinement_keeps_guides_and_evaluation_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,lesson=fixture(root);source=root/'v1';out=root/'v2'
            build(recipes,lesson,source,False)
            config=root/'refinements.json';refinement=json.loads((ROOT/'configs/helles-followup-refinements.json').read_text())
            refinement['extra_pairs']=[{'question':'Which settings are actually known?', 'answer':'Primary is specified.'}]
            write_json(config,refinement)
            report=refine_followups(source,lesson,config,out)
            self.assertEqual(report['train_examples'],47)
            self.assertEqual(refine_followups(source,lesson,config,out),report)
            for name in ['development.jsonl','final.jsonl','dialogue.json','validation.jsonl']:
                self.assertEqual((source/name).read_bytes(),(out/name).read_bytes())
            def guides(path):return [r['messages'][-1]['content'] for r in read_jsonl(path/'train.jsonl') if r['task']=='complete_guide']
            self.assertEqual(guides(source),guides(out))
            lesson_data=json.loads(config.read_text())
            target=next(r for r in read_jsonl(out/'train.jsonl') if r['task']=='followup' and 'grain-rest' in r['messages'][-2]['content'])
            self.assertEqual(target['messages'][-1]['content'],lesson_data['answers']['mash'])
            added=next(r for r in read_jsonl(out/'train.jsonl') if r['task']=='status_contrast')
            self.assertEqual(added['messages'][-2]['content'],'Which settings are actually known?')
            self.assertEqual(added['messages'][-1]['content'],'Primary is specified.')
            lesson_data['variants']['mash'].append(read_jsonl(source/'final.jsonl')[0]['messages'][1]['content'])
            write_json(config,lesson_data)
            with self.assertRaisesRegex(ValueError,'Evaluation question'):refine_followups(source,lesson,config,root/'leaked')
            (source/'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'Upstream focus file changed'):refine_followups(source,lesson,config,root/'changed')

    def test_focus_has_complete_guides_one_family_and_separate_question_wordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,config=fixture(root);out=root/'data'
            manifest=build(recipes,config,out,False)
            self.assertEqual(build(recipes,config,out,False),manifest)
            train=read_jsonl(out/'train.jsonl')
            self.assertEqual({r['family_id'] for r in train},{'one'})
            self.assertEqual(manifest['tasks']['complete_guide'],6)
            self.assertEqual(len(train),24)
            self.assertTrue(all('9.5 lb (4.31 kg) Pilsner malt' in r['messages'][-1]['content']
                                for r in train if r['task']=='complete_guide'))
            self.assertTrue(all(not any(c.isdigit() for c in r['messages'][0]['content']) for r in train))
            training_questions={m['content'] for r in train for m in r['messages'] if m['role']=='user'}
            for name in ['development.jsonl','final.jsonl']:
                self.assertFalse(training_questions&{r['messages'][1]['content'] for r in read_jsonl(out/name)})
            with (out/'train.jsonl').open('a') as stream:stream.write('{}\n')
            with self.assertRaisesRegex(ValueError,'Frozen focus files changed'):build(recipes,config,out,False)

    def test_source_change_heldout_recipe_and_copied_evaluation_question_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,config=fixture(root)
            rows=read_jsonl(recipes);rows[0]['provenance']['html_sha256']='changed';write_jsonl(recipes,rows)
            with self.assertRaisesRegex(ValueError,'source hash differs'):build(recipes,config,root/'out',False)
            recipes,config=fixture(root);rows=read_jsonl(recipes);rows[0]['split']='test';write_jsonl(recipes,rows)
            with self.assertRaisesRegex(ValueError,'training candidate'):build(recipes,config,root/'out',False)
            recipes,config=fixture(root);lesson=json.loads(config.read_text())
            lesson['development'][0]['question']=lesson['guide_questions'][0];write_json(config,lesson)
            with self.assertRaisesRegex(ValueError,'Evaluation question'):build(recipes,config,root/'out',False)

    def test_dialogue_resume_uses_generated_answers_never_reference_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol=root/'dialogue.json';out=root/'answers.json'
            write_json(protocol,{'scope':'one recipe','system_prompt':'Teach this recipe.',
                'turns':[{'question':'Teach me.','reference':'REFERENCE GUIDE','max_new_tokens':10},
                         {'question':'Then what?','reference':'REFERENCE FOLLOWUP','max_new_tokens':10}]})
            prompts=[]
            def generated(model,tokenizer,prompt,*args):
                prompts.append(prompt)
                if len(prompts)==2:raise RuntimeError('simulated interruption')
                return {'text':'ACTUAL GENERATED ANSWER','generated_tokens':3,'hit_token_limit':False}
            with patch('beerllm.recipe_focus.load_model',return_value=(object(),object())),\
                 patch('beerllm.recipe_focus.fit_history',side_effect=lambda messages,*args:messages),\
                 patch('beerllm.recipe_focus.generate',side_effect=generated):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct')
                result=run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct')
            self.assertEqual(result['status'],'complete')
            second=result['turns'][1]['prompt']
            self.assertEqual(second[-2],{'role':'assistant','content':'ACTUAL GENERATED ANSWER'})
            self.assertNotIn('REFERENCE GUIDE',str(second))
            with patch('beerllm.recipe_focus.load_model',side_effect=AssertionError('must not reload')):
                self.assertEqual(run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct'),result)
            changed=json.loads(protocol.read_text());changed['turns'][0]['question']='Changed';write_json(protocol,changed)
            with self.assertRaisesRegex(ValueError,'signature differs'):run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct')


if __name__=='__main__':unittest.main()
