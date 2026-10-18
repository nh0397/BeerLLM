import unittest
from beerllm.aha_dataset import build, conversation


def record(title,url):
    return {'title':title,'source_url':url,'html_sha256':'abc','batch_yield':'20 L',
            'ingredients_text':['4 kg pale malt'], 'directions_text':['Mash at 65°C for 60 minutes.'],
            'audit':{'missing_stage_mentions':['packaging']}}


class AhaDatasetTests(unittest.TestCase):
    def test_natural_prompt_and_original_instruction_target(self):
        row=conversation(record('Ale','https://example.com/ale'))
        self.assertEqual([m['role'] for m in row['messages']],['system','user','assistant'])
        self.assertNotIn('Recipe record:',row['messages'][1]['content'])
        self.assertIn('4 kg pale malt',row['messages'][2]['content'])
        self.assertIn('Mash at 65°C for 60 minutes.',row['messages'][2]['content'])
        self.assertIn('packaging',row['messages'][2]['content'])

    def test_variants_stay_together_and_splits_are_disjoint(self):
        rows=[record('Ale','a'),record('Ale V2','b'),record('Stout','c'),record('Lager','d')]
        splits=build(rows)
        where={row['source_url']:split for split,batch in splits.items() for row in batch}
        self.assertEqual(where['a'],where['b'])
        self.assertTrue(all(splits.values()))
        self.assertEqual(sum(map(len,splits.values())),4)

    def test_small_pilot_does_not_fake_splits(self):
        with self.assertRaisesRegex(ValueError,'three'):
            build([record('Ale','a')])


if __name__=='__main__': unittest.main()
