import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from beerllm.common import ROOT,read_jsonl,write_json,write_jsonl
from beerllm.recipe_focus import build,run_dialogue,refine_followups,package_adapter,sha
from beerllm.chat import adapter_decoding
from beerllm.train import replay_short_facts


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
    def test_retained_training_preserves_old_questions_and_latest_answer_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,lesson=fixture(root);source=root/'v1'
            build(recipes,lesson,source,False)
            config=root/'refine.json'
            base={'answers':{},'variants':{},'conversation_keys':[]}
            write_json(config,dict(base,extra_pairs=[{'question':'An older kettle wording?', 'answer':'Original answer.'},
                                                    {'question':'Retain this older question?', 'answer':'Keep this answer.'}]))
            first=root/'first';refine_followups(source,lesson,config,first)
            write_json(config,dict(base,extra_pairs=[{'question':'An older kettle wording?', 'answer':'Updated answer.'}]))
            second=root/'second';refine_followups(source,lesson,config,second)
            snapshots={p:(p/'train.jsonl').read_bytes() for p in [first,second]}
            write_json(config,base);out=root/'combined'
            result=refine_followups(source,lesson,config,out,[first,second])
            rows=read_jsonl(out/'train.jsonl')
            self.assertEqual(len(rows),26)
            selected=[r for r in rows if r['messages'][-2]['content']=='An older kettle wording?']
            self.assertEqual(len(selected),1)
            self.assertEqual(selected[0]['messages'][-1]['content'],'Updated answer.')
            self.assertTrue(any(r['messages'][-2]['content']=='Retain this older question?' for r in rows))
            self.assertEqual(result['retained_training_manifests_sha256'],[sha(first/'manifest.json'),sha(second/'manifest.json')])
            self.assertEqual(refine_followups(source,lesson,config,out,[first,second]),result)
            for parent,content in snapshots.items():self.assertEqual((parent/'train.jsonl').read_bytes(),content)
            for name in ['development.jsonl','final.jsonl','dialogue.json','validation.jsonl']:
                self.assertEqual((source/name).read_bytes(),(out/name).read_bytes())

    def test_retained_training_rejects_tampering_and_a_different_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,lesson=fixture(root);source=root/'v1'
            build(recipes,lesson,source,False)
            config=root/'refine.json';write_json(config,{'answers':{},'variants':{},'conversation_keys':[]})
            parent=root/'parent';refine_followups(source,lesson,config,parent)
            original=(parent/'train.jsonl').read_bytes();(parent/'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'Retained training file changed'):
                refine_followups(source,lesson,config,root/'tampered',[parent])
            (parent/'train.jsonl').write_bytes(original)
            recipe=read_jsonl(parent/'recipe.jsonl');recipe[0]['recipe_id']='different'
            write_jsonl(parent/'recipe.jsonl',recipe)
            manifest=json.loads((parent/'manifest.json').read_text());manifest['files_sha256']['recipe.jsonl']=sha(parent/'recipe.jsonl')
            write_json(parent/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'same frozen recipe'):
                refine_followups(source,lesson,config,root/'mixed',[parent])

    def test_retention_does_not_bypass_reserved_question_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,lesson=fixture(root);source=root/'v1'
            build(recipes,lesson,source,False)
            config=root/'refine.json';write_json(config,{'answers':{},'variants':{},'conversation_keys':[]})
            parent=root/'parent';refine_followups(source,lesson,config,parent)
            rows=read_jsonl(parent/'train.jsonl');leak=dict(rows[0]);leak['messages']=read_jsonl(source/'final.jsonl')[0]['messages'];rows.append(leak)
            write_jsonl(parent/'train.jsonl',rows)
            manifest=json.loads((parent/'manifest.json').read_text());manifest['files_sha256']['train.jsonl']=sha(parent/'train.jsonl')
            write_json(parent/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'Evaluation question copied'):
                refine_followups(source,lesson,config,root/'leaked',[parent])

    def test_scope_pairs_teach_different_starting_points_without_copying_evaluation_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,lesson=fixture(root);source=root/'v1';out=root/'scope'
            build(recipes,lesson,source,False)
            path=ROOT/'configs/helles-scope-refinements.json'
            config=json.loads(path.read_text());pairs=[x for x in config['extra_pairs'] if 'scope_pair' in x]
            self.assertEqual(len(pairs),16)
            for pair in range(1,9):
                sides=[x for x in pairs if x['scope_pair']==pair]
                self.assertEqual({x['scope_start'] for x in sides},{'boil','primary'})
                for item in sides:
                    self.assertIn('40°F',item['answer'])
                    self.assertEqual('50°F' in item['answer'],item['scope_start']=='boil')
            report=refine_followups(source,lesson,path,out);rows=read_jsonl(out/'train.jsonl')
            self.assertEqual(report['train_examples'],62)
            self.assertEqual(len(replay_short_facts(rows,3)),150)
            questions={m['content'] for r in rows for m in r['messages'] if m['role']=='user'}
            reserved={r['messages'][1]['content'] for r in read_jsonl(ROOT/'configs/helles-explanation-dev.jsonl')}
            reserved.update(x['question'] for x in json.loads((ROOT/'configs/helles-natural-dialogue.json').read_text())['turns'])
            self.assertFalse(questions&reserved)
            self.assertEqual((source/'final.jsonl').read_bytes(),(out/'final.jsonl').read_bytes())

    def test_package_binds_decoding_to_unchanged_weights_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir();out=root/'release'
            (source/'adapter_model.safetensors').write_bytes(b'unchanged trained weights')
            model='Qwen/Qwen2.5-1.5B-Instruct';revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
            write_json(source/'adapter_config.json',{'base_model_name_or_path':model,'revision':revision,'r':16,'lora_alpha':32})
            write_json(source/'beerllm.json',{'format_version':1,'model_id':model,'model_revision':revision,'system_prompt':'Teach this recipe.'})
            policy=root/'decoding.json';write_json(policy,{'num_beams':2,'length_penalty':1.0})
            report=package_adapter(source,out,policy)
            self.assertEqual(package_adapter(source,out,policy),report)
            self.assertEqual((source/'adapter_model.safetensors').read_bytes(),(out/'adapter_model.safetensors').read_bytes())
            self.assertEqual((source/'adapter_config.json').read_bytes(),(out/'adapter_config.json').read_bytes())
            self.assertIsNone(adapter_decoding(source))
            self.assertEqual(adapter_decoding(out),{'num_beams':2,'length_penalty':1.0})
            with self.assertRaisesRegex(ValueError,'outside the source'):
                package_adapter(source,source/'nested',policy)
            (out/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen package'):
                package_adapter(source,out,policy)

    def test_concept_followups_replay_without_leaking_reserved_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,lesson=fixture(root);source=root/'v1';out=root/'teacher'
            build(recipes,lesson,source,False)
            config=ROOT/'configs/helles-teacher-refinements.json'
            report=refine_followups(source,lesson,config,out)
            rows=read_jsonl(out/'train.jsonl');replayed=replay_short_facts(rows,3)
            self.assertEqual(report['train_examples'],132)
            self.assertEqual(len(replayed),332)
            concept=json.loads(config.read_text())['extra_pairs'][-3]
            self.assertEqual(sum(r['messages'][-2]['content']==concept['question'] for r in replayed),3)
            conversation=next(r for r in rows if r['messages'][-2]['content']==
                              json.loads(config.read_text())['extra_pairs'][-1]['question'])
            guide_row=next(r for r in rows if r['task']=='complete_guide')
            self.assertEqual(conversation['messages'][2]['content'],guide_row['messages'][-1]['content'])
            self.assertEqual(conversation['task'],'conversation_variant')
            questions={m['content'] for r in rows for m in r['messages'] if m['role']=='user'}
            reserved={r['messages'][1]['content'] for r in read_jsonl(ROOT/'configs/helles-explanation-dev.jsonl')}
            reserved.update(x['question'] for x in json.loads((ROOT/'configs/helles-natural-dialogue.json').read_text())['turns'])
            self.assertFalse(questions&reserved)
            for name in ['development.jsonl','final.jsonl','dialogue.json','validation.jsonl']:
                self.assertEqual((source/name).read_bytes(),(out/name).read_bytes())

    def test_balanced_curriculum_preserves_source_answers_and_reserves_new_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipes,lesson=fixture(root);source=root/'v1';out=root/'balanced'
            build(recipes,lesson,source,False)
            config=root/'balanced.json'
            refinement=json.loads((ROOT/'configs/helles-balanced-refinements.json').read_text())
            write_json(config,refinement)
            report=refine_followups(source,lesson,config,out)
            self.assertEqual(report['train_examples'],150)
            rows=read_jsonl(out/'train.jsonl')
            guides=[r for r in rows if r['task']=='complete_guide']
            self.assertEqual(len(guides),18)
            self.assertEqual(len({r['messages'][-1]['content'] for r in guides}),1)
            conversation=next(r for r in rows if r['task']=='conversation_variant')
            self.assertEqual(conversation['messages'][2]['content'],guides[0]['messages'][-1]['content'])
            old=read_jsonl(source/'final.jsonl');new=read_jsonl(out/'confirmation.jsonl')
            self.assertEqual([r['messages'][-1] for r in old],[r['messages'][-1] for r in new])
            self.assertTrue(all(a['messages'][1]!=b['messages'][1] for a,b in zip(old,new)))
            self.assertIn('confirmation.jsonl',report['files_sha256'])
            refinement['confirmation_questions']['final-guide']=guides[0]['messages'][1]['content']
            write_json(config,refinement)
            with self.assertRaisesRegex(ValueError,'Confirmation question duplicated'):
                refine_followups(source,lesson,config,root/'leaked')

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
            policy={'num_beams':2,'length_penalty':1.0}
            def generated(model,tokenizer,prompt,*args,**kwargs):
                self.assertEqual(kwargs['decoding'],policy)
                prompts.append(prompt)
                if len(prompts)==2:raise RuntimeError('simulated interruption')
                return {'text':'ACTUAL GENERATED ANSWER','generated_tokens':3,'hit_token_limit':False}
            with patch('beerllm.recipe_focus.load_model',return_value=(object(),object())),\
                 patch('beerllm.recipe_focus.fit_history',side_effect=lambda messages,*args:messages),\
                 patch('beerllm.recipe_focus.generate',side_effect=generated):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct',decoding=policy)
                result=run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct',decoding=policy)
            self.assertEqual(result['status'],'complete')
            second=result['turns'][1]['prompt']
            self.assertEqual(second[-2],{'role':'assistant','content':'ACTUAL GENERATED ANSWER'})
            self.assertNotIn('REFERENCE GUIDE',str(second))
            self.assertEqual(result['signature']['decoding']['num_beams'],2)
            with patch('beerllm.recipe_focus.load_model',side_effect=AssertionError('must not reload')):
                self.assertEqual(run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct',decoding=policy),result)
            with self.assertRaisesRegex(ValueError,'signature differs'):
                run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct')
            changed=json.loads(protocol.read_text());changed['turns'][0]['question']='Changed';write_json(protocol,changed)
            with self.assertRaisesRegex(ValueError,'signature differs'):run_dialogue(protocol,out,'Qwen/Qwen2.5-1.5B-Instruct',decoding=policy)


if __name__=='__main__':unittest.main()
