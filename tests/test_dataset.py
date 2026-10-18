import unittest
from beerllm.dataset import build, family_groups, guide


def recipe(i=1):
    return {'id':i,'name':'Test Ale','volume':{'value':20,'unit':'litres'},
            'boil_volume':{'value':25,'unit':'litres'},'abv':5,
            'ingredients':{'malt':[{'name':'Pale','amount':{'value':4,'unit':'kilograms'}}],
                           'hops':[{'name':'Cascade','amount':{'value':20,'unit':'grams'},'add':'middle'}],
                           'yeast':'US-05'},
            'method':{'mash_temp':[{'temp':{'value':65,'unit':'celsius'},'duration':60}],
                      'fermentation':{'temp':{'value':19,'unit':'celsius'}}},'brewers_tips':'Use fresh hops.'}


class DatasetTests(unittest.TestCase):
    def test_missing_mash_rejected(self):
        r=recipe(); r['method']['mash_temp'][0]['duration']=None
        _,guides,rejected=build([r])
        self.assertFalse(guides)
        self.assertIn('mash_schedule',rejected[0]['reasons'])

    def test_guides_keep_source_quantities_and_unknown_timings(self):
        text=guide(recipe())
        self.assertIn('20 grams',text)
        self.assertIn('hop addition label: middle',text)
        self.assertIn('does not specify total boil duration',text)
        self.assertNotIn('30 minutes',text)

    def test_family_and_tasks_stay_in_one_split(self):
        a,b=recipe(),recipe(2)
        b['name']='Test Ale V2'
        b['ingredients']['malt'][0]['amount']['value']=5
        self.assertEqual(family_groups([a,b])[1],family_groups([a,b])[2])
        splits,guides,_=build([a,b])
        self.assertEqual(guides[0]['split'],guides[1]['split'])
        for split,rows in splits.items():
            if split!='train': self.assertTrue(all(r['task']=='grounded_guide' for r in rows))

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            build([recipe(),recipe()])

    def test_special_process_rejected(self):
        r=recipe();r['name']='Barrel Aged Sour'
        _,guides,rejected=build([r])
        self.assertFalse(guides)
        self.assertIn('special_process_requires_review',rejected[0]['reasons'])


if __name__=='__main__': unittest.main()
