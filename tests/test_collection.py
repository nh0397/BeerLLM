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



class CollectionStorageTests(unittest.TestCase):
    def test_exact_duplicates_keep_both_provenances_and_resume_status(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from beerllm import collection
        from beerllm.common import ROOT
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            out=Path(temp)
            with patch.object(collection,'OUT',out),patch.object(collection,'DB',out/'queue.sqlite3'):
                db=collection.connection()
                base={'title':'Ale','batch_yield':'20 L','ingredients_text':['4 kg pale malt'],
                      'directions_text':['Mash at 65 C.'],'process_text':None,'notes':None}
                for source,url in [('aha','https://example.com/a'),('brewersfriend','https://example.com/b')]:
                    db.execute('INSERT INTO queue(url,source,source_id) VALUES(?,?,?)',(url,source,'1'))
                    collection.save_record(db,source,url,base.copy())
                self.assertEqual(db.execute("SELECT count(*) FROM queue WHERE status='pending'").fetchone()[0],0)
                db.close()
                collection.export_snapshot()
                rows=[json.loads(line) for line in (out/'unique_recipes.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),1)
                self.assertEqual(len(rows[0]['sources']),2)
                report=json.loads((out/'report.json').read_text())
                self.assertEqual(report['exact_duplicates_merged'],1)


class ThrottleTests(unittest.TestCase):
    def test_429_marks_source_blocked_and_honors_cooldown(self):
        from unittest.mock import Mock, patch
        import requests
        from beerllm.collection import Client
        client=Client.__new__(Client)
        client.origin='https://example.com'
        client.robots=Mock()
        client.robots.can_fetch.return_value=True
        client.delay=0
        client.last=0
        client.blocked=False
        client.retry_seconds=3600
        response=requests.Response()
        response.status_code=429
        response.url='https://example.com/recipe'
        response.headers['Retry-After']='7200'
        client.session=Mock()
        client.session.get.return_value=response
        with patch('time.sleep'):
            with self.assertRaises(requests.HTTPError):
                client.get(response.url)
        self.assertTrue(client.blocked)
        self.assertEqual(client.retry_seconds,7200)
        with self.assertRaisesRegex(ValueError,'paused'):
            client.get(response.url)
        self.assertEqual(client.session.get.call_count,1)



class TotalLimitTests(unittest.TestCase):
    def test_total_cap_is_shared_by_concurrent_sources(self):
        import json
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from beerllm import collection
        from beerllm.common import ROOT
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            out=Path(temp)
            with patch.object(collection,'OUT',out),patch.object(collection,'DB',out/'queue.sqlite3'):
                db=collection.connection()
                db.close()
                def save(source):
                    db=collection.connection()
                    url='https://example.com/'+source
                    db.execute('INSERT INTO queue(url,source,source_id) VALUES(?,?,?)',(url,source,'1'))
                    result=collection.save_record(db,source,url,{'title':source},max_recipes=1)
                    db.close()
                    return result
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results=list(pool.map(save,['aha','brewersfriend']))
                self.assertEqual(sum(results),1)
                db=collection.connection()
                self.assertEqual(db.execute('SELECT count(*) FROM recipes').fetchone()[0],1)
                self.assertEqual(db.execute("SELECT count(*) FROM queue WHERE status='pending'").fetchone()[0],1)
                row=db.execute('SELECT source,url,data FROM recipes').fetchone()
                self.assertTrue(collection.save_record(db,row[0],row[1],json.loads(row[2]),max_recipes=1))
                db.close()
                self.assertTrue(collection.at_capacity(1))


if __name__=='__main__': unittest.main()
