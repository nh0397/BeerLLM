import io
import json
import unittest
from unittest.mock import patch
from scripts.fetch_recipes import audit, download


class RecipeTests(unittest.TestCase):
    def test_missing_mash_duration_and_prose_require_review(self):
        result = audit({'method': {'mash_temp': [{'temp': {'value': 65, 'unit': 'celsius'}, 'duration': None}]},
                        'brewers_tips': 'Ferment and package as usual.'})
        self.assertIn('mash_schedule', result['missing_fields'])
        self.assertIn('packaging', result['needs_instruction_review'])
        self.assertEqual(result['status'], 'needs_review')

    def test_pagination_continues_until_empty(self):
        pages = [io.BytesIO(json.dumps(x).encode()) for x in [[{'id': 1}], [{'id': 2}], []]]
        with patch('urllib.request.urlopen', side_effect=pages) as request:
            self.assertEqual([r['id'] for r in download()], [1, 2])
            self.assertEqual(request.call_count, 3)

    def test_repeated_page_fails(self):
        pages = [io.BytesIO(b'[{"id": 1}]') for _ in range(2)]
        with patch('urllib.request.urlopen', side_effect=pages):
            with self.assertRaisesRegex(ValueError, 'Repeated recipe'):
                download()


if __name__ == '__main__':
    unittest.main()
