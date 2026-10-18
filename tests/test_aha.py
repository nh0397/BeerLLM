import unittest
from beerllm.aha import parse_recipe, recipe_url

HTML = '''<h1>Test Ale</h1><div class="ingredients"><div><h3>Ingredients</h3>
<ul><li>MALT</li><li>4 kg pale malt</li><li>20 g Cascade @ 60 min</li></ul></div>
<div><p><strong>Yield:</strong> 20 L</p></div></div>
<div itemprop="recipeInstructions"><p>Mash at 65°C for 60 minutes.</p>
<p>Boil, chill, then pitch yeast.</p><p>Check fermentation before bottling.</p></div>'''


class AhaTests(unittest.TestCase):
    def test_preserves_order_and_original_units(self):
        r = parse_recipe(HTML,'https://homebrewersassociation.org/homebrew-recipe/test/')
        self.assertEqual(r['batch_yield'],'20 L')
        self.assertEqual(r['ingredients_text'][1],'4 kg pale malt')
        self.assertEqual(r['directions_text'][0],'Mash at 65°C for 60 minutes.')
        self.assertNotIn('Yield: 20 L',r['ingredients_text'])
        self.assertEqual(r['audit']['status'],'needs_review')

    def test_missing_directions_rejected(self):
        with self.assertRaisesRegex(ValueError,'unavailable'):
            parse_recipe('<h1>Members only</h1>', 'https://example.com')

    def test_missing_stage_is_reported(self):
        r = parse_recipe(HTML.replace('Check fermentation before bottling.','Wait.'),'https://example.com')
        self.assertIn('packaging',r['audit']['missing_stage_mentions'])

    def test_url_is_limited_to_recipe_pages(self):
        with self.assertRaises(ValueError):
            recipe_url('https://example.com/homebrew-recipe/test/')
        with self.assertRaises(ValueError):
            recipe_url('https://homebrewersassociation.org/membership/')
        self.assertEqual(recipe_url('https://www.homebrewersassociation.org/homebrew-recipe/test?x=1'),
                         'https://homebrewersassociation.org/homebrew-recipe/test/')


if __name__ == '__main__':
    unittest.main()
