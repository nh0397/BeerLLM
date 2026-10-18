import unittest
from beerllm.collection import fingerprint, family, instructions, locations, parse_brewersfriend, source_id


class CollectionTests(unittest.TestCase):
    def test_sitemap_namespaces(self):
        tag,urls=locations('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://example.com/one</loc></url></urlset>')
        self.assertEqual(tag,'urlset')
        self.assertEqual(urls,['https://example.com/one'])

    def test_recipe_scope(self):
        self.assertEqual(source_id('brewersfriend','https://www.brewersfriend.com/homebrew/recipe/view/12/ale'),'12')
        self.assertIsNone(source_id('brewersfriend','https://www.brewersfriend.com/homebrew/recipe/export/12'))
        self.assertEqual(source_id('aha','https://homebrewersassociation.org/homebrew-recipe/ale/'),'ale')

    def test_duplicate_names_do_not_discard_process_variants(self):
        a={'title':'Ale','batch_yield':'20 L','ingredients_text':['1 kg malt'],
           'directions_text':['Mash at 65 C'],'process_text':None,'notes':None}
        b=a | {'title':'Ale copy'}
        self.assertEqual(fingerprint(a),fingerprint(b))
        c=a | {'directions_text':['Mash at 70 C']}
        self.assertNotEqual(fingerprint(a),fingerprint(c))
        d=a | {'ingredients_text':['2 kg malt']}
        self.assertNotEqual(fingerprint(a),fingerprint(d))
        self.assertEqual(family(a),family(a | {'title':'Ale V2'}))

    def test_instruction_sections_preserve_order(self):
        value=[{'@type':'HowToSection','itemListElement':[{'text':'Mash.'},{'text':'Boil.'}]},{'text':'Ferment.'}]
        self.assertEqual(instructions(value),['Mash.','Boil.','Ferment.'])

    def test_public_recipe_jsonld_and_full_process_text(self):
        html='''<script type="application/ld+json">{"@graph":[{"@type":"Recipe","name":"Ale","recipeIngredient":["4 kg malt"],"recipeInstructions":[{"text":"Mash at 65 C."}]}]}</script><textarea>HOME BREW RECIPE:\nTitle: Ale\nHOPS: 20 g Cascade, Boil for 60 min\nThis recipe has been published online at: https://example.com</textarea>'''
        r=parse_brewersfriend(html,'https://example.com')
        self.assertEqual(r['directions_text'],['Mash at 65 C.'])
        self.assertIn('20 g Cascade',r['process_text'])
        self.assertNotIn('Title: Ale',r['process_text'])
        self.assertEqual(r['status'],'needs_review')

    def test_unavailable_recipe_not_accepted(self):
        with self.assertRaisesRegex(ValueError,'No public recipe'):
            parse_brewersfriend('<h1>Login</h1>','https://example.com')


if __name__=='__main__': unittest.main()
