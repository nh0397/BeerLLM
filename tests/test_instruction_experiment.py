import copy
import json
from pathlib import Path
import tempfile
import unittest
from beerllm.benchmark import compare,pattern_checks,repetition,summarize
from beerllm.common import write_json
from beerllm.instruction_dataset import guide,lesson_rows


def recipe():
    return {'recipe_id':'r','family_id':'f','split':'train','title':'Example Ale',
            'source_url':'https://example.com/ale','batch_yield_original':'5 US gal (19 L)',
            'provenance':{'html_sha256':'source-hash'},'audit':{'tier':'instruction_candidate'},
            'ingredients':[{'source_text':'2 oz Cascade (60 min)'},{'source_text':'US-05 yeast'}],
            'directions':['Mash at 65°C for 60 minutes. Boil for 60 minutes.']}


class InstructionExperimentTests(unittest.TestCase):
    def test_authored_guides_preserve_source_ingredients_and_mark_general_guidance(self):
        r=recipe()
        answer=guide(r,['Mash as specified.'],'No sugar dose is specified.')
        self.assertIn('2 oz Cascade (60 min)',answer)
        self.assertIn('1. Mash as specified.',answer)
        self.assertIn('No sugar dose is specified.',answer)
        self.assertIn('General brewing checks, separate from the recipe:',answer)
        self.assertEqual(r['directions'],['Mash at 65°C for 60 minutes. Boil for 60 minutes.'])

    def test_teaching_tasks_cannot_import_heldout_families(self):
        lesson={'steps':['Mash as specified.'],'gaps':'No sugar dose.',
                'followups':[['What is missing?','The sugar dose.']]}
        r=recipe();rows=lesson_rows(r,lesson)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(x['family_id']=='f' for x in rows))
        self.assertNotIn('Source directions',rows[0]['messages'][1]['content'])
        self.assertIn('Source directions',rows[1]['messages'][1]['content'])
        self.assertTrue(all(x['review_status']=='source_checked_experiment_not_brewer_reviewed' for x in rows))
        r['split']='test'
        with self.assertRaisesRegex(ValueError,'training-partition'):
            lesson_rows(r,lesson)

    def test_perplexity_uses_token_weighted_answer_loss(self):
        def item(loss,tokens):
            return {'category':'heldout','answer_tokens':tokens,'total_answer_nll':loss*tokens,
                    'pattern_checks':[{'passed':True}],'hit_token_limit':False,
                    'generation_seconds':1,'repeated_4gram_fraction':0}
        score=summarize([item(1,1),item(3,3)])['heldout']
        self.assertAlmostEqual(score['answer_token_cross_entropy'],2.5)
        self.assertAlmostEqual(score['conditional_answer_perplexity'],12.1824939607)
        self.assertEqual(score['answer_tokens'],4)

    def test_pattern_checks_do_not_reward_wrong_quantity_alone(self):
        checks=[['dose',r'1\.2\s*oz\s+Centennial']]
        self.assertFalse(pattern_checks('Add 5 oz Centennial.',checks)[0]['passed'])
        self.assertTrue(pattern_checks('Add 1.2 oz Centennial.',checks)[0]['passed'])
        self.assertEqual(repetition('short answer'),0)
        self.assertGreater(repetition('one two three four one two three four one two three four'),0)

    def test_comparison_rejects_changed_prompts_or_decoding(self):
        signature={k:'same' for k in ['suite_sha256','model_id','model_revision','decoding','evaluator_sha256']}
        before={'status':'complete','signature':dict(signature,mode='base'),
                'summary':{},'examples':[{'case_id':'one'}]}
        after=copy.deepcopy(before);after['signature']['mode']='adapter';after['limitations']=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'before.json';b=root/'after.json';out=root/'comparison.json'
            write_json(a,before);write_json(b,after)
            self.assertEqual(compare(a,b,out)['status'],'complete')
            after['signature']['suite_sha256']='different';write_json(b,after)
            with self.assertRaisesRegex(ValueError,'Mismatched before/after suite_sha256'):
                compare(a,b,out)


if __name__=='__main__': unittest.main()
