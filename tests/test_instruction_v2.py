import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from beerllm.common import MODEL_ID, MODEL_REVISION, model_revision
from beerllm.common import write_jsonl
from beerllm.chat import curated_context
from beerllm.instruction_v2 import ensure_disjoint, expanded_rows, verify_source
from beerllm.train import encode_example
from test_training import TinyTokenizer
from test_instruction_experiment import recipe


class InstructionV2Tests(unittest.TestCase):
    def test_explicit_source_selection_preserves_notes_and_rejects_unreviewed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recipes.jsonl';r=recipe()
            write_jsonl(path,[r])
            selected,context=curated_context(path,'example ale')
            self.assertEqual(selected['recipe_id'],r['recipe_id'])
            self.assertIn('Mash at 65°C for 60 minutes.',context)
            self.assertIn('2 oz Cascade (60 min)',context)
            r['audit']['tier']='quarantine_conflict';write_jsonl(path,[r])
            with self.assertRaisesRegex(ValueError,'source review'): curated_context(path,'Example Ale')
            r['audit']['tier']='instruction_candidate';write_jsonl(path,[r,r])
            with self.assertRaisesRegex(ValueError,'found 2'): curated_context(path,'Example Ale')

    def test_pins_models_and_rejects_mutable_or_missing_revision(self):
        self.assertEqual(model_revision(MODEL_ID), MODEL_REVISION)
        self.assertEqual(model_revision('another/model','a'*40),'a'*40)
        for revision in [None,'main','xyz'*14]:
            with self.assertRaises(ValueError): model_revision('another/model',revision)

    def test_family_guard_rejects_title_changes_in_same_family(self):
        training=[{'family_id':'same','title':'Original'}]
        with self.assertRaisesRegex(ValueError,'family leaks'):
            ensure_disjoint(training,[{'family_id':'same','title':'Clone'}])
        ensure_disjoint(training,[{'family_id':'different'}])

    def test_followup_history_is_context_not_supervised_target(self):
        lesson={'steps':['Mash at 65°C for 60 minutes.'],'gaps':'No priming dose.',
                'followups':[['Which mash?','Mash at 65°C for 60 minutes.'],['What is missing?','No priming dose.']]}
        rows=expanded_rows(recipe(),lesson,[['Describe the mash.','Give the mash settings.'],['Any missing dose?','What should I confirm?']])
        self.assertEqual(len(rows),12)
        conversation=next(r for r in rows if r['task']=='conversation_followup')
        self.assertEqual([m['role'] for m in conversation['messages']],['system','user','assistant','user','assistant'])
        item=encode_example(TinyTokenizer(),conversation,10000)
        supervised=''.join(chr(i) for i in item['labels'] if i!=-100)
        self.assertEqual(supervised,conversation['messages'][-1]['content']+'<end>')
        bad=recipe();bad['split']='test'
        with self.assertRaisesRegex(ValueError,'training-partition'):
            expanded_rows(bad,lesson,[['A','B'],['C','D']])

    def test_source_verification_detects_changed_html_and_reextracted_facts(self):
        html='<h1>Example</h1><div class="ingredients"><ul><li>4 kg malt</li></ul></div><div itemprop="recipeInstructions"><p>Mash at 65°C.</p></div>'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'source.html').write_text(html)
            r=recipe();r['provenance']={'raw_html':'source.html','html_sha256':hashlib.sha256(html.encode()).hexdigest()}
            r['ingredients']=[{'source_text':'4 kg malt'}];r['directions']=['Mash at 65°C.']
            with patch('beerllm.instruction_v2.ROOT',root):
                self.assertTrue(verify_source(r)['ingredients_and_directions_match'])
                r['directions']=['Mash at 75°C.']
                with self.assertRaisesRegex(ValueError,'Re-extracted source differs'): verify_source(r)
                (root/'source.html').write_text(html+'changed')
                with self.assertRaisesRegex(ValueError,'HTML hash differs'): verify_source(r)


if __name__=='__main__': unittest.main()
