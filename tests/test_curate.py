import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from beerllm.common import read_jsonl,write_jsonl
from beerllm.curate import assign_families,batch_liters,build_release,clean,conversation,mass,nonbeer,repair_ingredients,temperature_conflicts,text


def recipe(identifier='one',source='aha'):
    return {'content_id':identifier,'title':'Test Ale','source':source,
            'source_url':'https://example.com/'+identifier,'sources':[],
            'batch_yield':'5 gallons (19 L)',
            'ingredients_text':['4 kg Pale Malt','20 g Cascade hops','US-05 yeast'],
            'directions_text':['Mash at 65°C for 60 minutes. Boil for 60 minutes.',
                               'Chill, pitch yeast and ferment at 20°C. Keg when fermentation is complete.'],
            'process_text':None,'notes':None}


class CurationTests(unittest.TestCase):
    def test_preserve_beer_styles_while_excluding_other_drinks(self):
        for title in ['Barley Wine','English barleywine','Wheatwine','Chai Tea Brown Ale','Anchor Steam Clone']:
            self.assertFalse(nonbeer({'title':title,'style':None},[]),title)
        for title in ['Apple Cider','Some kind of Mead','Basic Kombucha','Chai-Barley Tea']:
            self.assertTrue(nonbeer({'title':title,'style':None},[]),title)

    def test_metric_volume_and_explicit_us_only(self):
        self.assertEqual(batch_liters('6 Gallons (22.7 L)')['value'],22.7)
        self.assertAlmostEqual(batch_liters('5 US gal')['value'],18.92705892)
        self.assertIsNone(batch_liters('5 gallons'))
        self.assertIsNone(batch_liters({'value':-2,'unit':'litres'}))

    def test_fractions_and_mass_preserve_source_value(self):
        self.assertEqual(text('½ oz Cascade'),'1/2 oz Cascade')
        self.assertAlmostEqual(mass(text('1½ lb Pale Malt'))['value'],680.388555)
        self.assertEqual(mass('2 kg Pale Malt')['value'],2000)
        self.assertIsNone(mass('4.9% a.a.'))

    def test_join_split_hop_suffix_without_losing_provenance(self):
        rows,repairs=repair_ingredients(['HOPS','2 oz Fuggles hops,','4.9% a.a. (0 min.)'])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['source_line_indices'],[1,2])
        self.assertIn('0 min.',rows[0]['source_text'])
        self.assertEqual(repairs,['joined_wrapped_alpha_acid_line'])

    def test_recover_yeast_and_other_ingredients_from_public_text(self):
        r=recipe(source='brewersfriend')
        r['process_text']='FERMENTABLES:\n4 kg - Pale malt\nHOPS:\n20 g - Cascade, Boil for 60 min\nOTHER INGREDIENTS:\n1 tsp - Nutrient\nYEAST:\nUS-05\nFermentation Temp: 20 C\nPRIMING:\nCO2 Level: 2.5 Volumes'
        result=clean(r)
        lines=[x['source_text'] for x in result['ingredients']]
        self.assertIn('US-05',lines)
        self.assertIn('1 tsp - Nutrient',lines)
        self.assertIn('20 g - Cascade, Boil for 60 min',lines)
        self.assertNotIn('Fermentation Temp: 20 C',lines)

    def test_temperature_conflicts_are_quarantined_not_corrected(self):
        r=recipe()
        r['directions_text'].append('Store at 40-50°F (9-10°C).')
        result=clean(r)
        self.assertEqual(result['audit']['tier'],'quarantine_conflict')
        self.assertIn('40-50°F (9-10°C)',result['directions'][-1])
        self.assertFalse(temperature_conflicts('Mash at 152°F (67°C).'))

    def test_missing_steps_are_not_generated(self):
        r=recipe();r['directions_text']=['Mash at 65°C for 60 minutes.']
        result=clean(r)
        self.assertEqual(result['audit']['tier'],'needs_enrichment')
        self.assertIn('packaging',result['audit']['missing_stages'])
        self.assertEqual(result['directions'],r['directions_text'])

    def test_grouping_keeps_variants_and_task_partitions_together(self):
        a=clean(recipe('a'));r=recipe('b');r['title']='Test Ale V2';r['directions_text'][0]='Mash at 66°C for 60 minutes. Boil 90 minutes.'
        b=clean(r)
        self.assertNotEqual(a['clean_content_hash'],b['clean_content_hash'])
        assign_families([a,b])
        self.assertEqual(a['family_id'],b['family_id'])
        self.assertEqual(a['split'],b['split'])
        row=conversation(a)
        self.assertNotIn('Recipe record:',row['messages'][1]['content'])
        self.assertEqual([m['role'] for m in row['messages']],['system','user','assistant'])

    def test_snapshot_and_review_metadata_survive_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.jsonl';output=root/'release'
            r=recipe();r['html_sha256']='html-hash';r['rights_status']='unverified'
            write_jsonl(source,[r])
            original=source.read_bytes()
            report=build_release(source,output)
            self.assertEqual((output/'source_snapshot.jsonl').read_bytes(),original)
            self.assertEqual(report['input_sha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual(report['approved_training_examples'],0)
            self.assertEqual(report['conversation_candidates'],1)
            row=read_jsonl(output/'recipes.jsonl')[0]
            self.assertEqual(row['provenance']['html_sha256'],'html-hash')
            self.assertEqual(row['provenance']['rights_status'],'unverified')
            self.assertFalse(row['audit']['expert_reviewed'])
            # A release directory cannot silently change to a different source corpus.
            write_jsonl(source,[recipe('other')])
            with self.assertRaisesRegex(ValueError,'different source snapshot'):
                build_release(source,output)
            self.assertEqual((output/'source_snapshot.jsonl').read_bytes(),original)
            self.assertEqual(json.loads((output/'manifest.json').read_text()),report)


if __name__=='__main__': unittest.main()
