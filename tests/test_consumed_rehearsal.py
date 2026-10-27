import json
from pathlib import Path
import tempfile
import unittest

from beerllm.common import read_jsonl, write_json, write_jsonl
from beerllm.recipe_focus import sha
from scripts.prepare_helles_rehearsal import prepare


class ConsumedRehearsalTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        system = {'role': 'system', 'content': 'Teach one recipe.'}
        row = {'task': 'complete_guide', 'messages': [system,
               {'role': 'user', 'content': 'Original practice'},
               {'role': 'assistant', 'content': 'Original answer'}]}
        for name in ['train.jsonl', 'validation.jsonl', 'recipe.jsonl']:
            write_jsonl(source / name, [row])
        write_json(source / 'manifest.json', {
            'files_sha256': {p.name: sha(p) for p in source.iterdir()}})
        cases = root / 'cases.jsonl'
        write_jsonl(cases, [dict(row, case_id='used-case', category='training_fit',
            messages=[system, {'role': 'user', 'content': 'Already inspected question'},
                      {'role': 'assistant', 'content': 'Corrected answer'}])])
        dialogue = root / 'dialogue.json'
        write_json(dialogue, {'system_prompt': system['content'], 'turns': [
            {'question': 'Consumed turn one', 'reference': 'Answer one'},
            {'question': 'Consumed turn two', 'reference': 'Answer two'}]})
        fresh = root / 'fresh.jsonl'
        write_jsonl(fresh, [{'messages': [system, {'role': 'user', 'content': 'Untouched wording'}]}])
        fresh_chat = root / 'fresh-chat.json'
        write_json(fresh_chat, {'turns': [{'question': 'Untouched conversation'}]})
        return source, cases, dialogue, fresh, fresh_chat

    def test_promotion_is_explicit_and_retains_history_and_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.fixture(root)
            snapshots = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            out = root / 'out'
            self.assertEqual(prepare(*inputs, out), 4)
            rows = read_jsonl(out / 'train.jsonl')
            self.assertEqual(rows[-1]['messages'][-3]['content'], 'Answer one')
            self.assertEqual(rows[-1]['messages'][-1]['content'], 'Answer two')
            self.assertIn('not held-out', json.loads((out / 'manifest.json').read_text())['scope'])
            self.assertFalse((out / 'final.jsonl').exists())
            for path, content in snapshots.items():
                self.assertEqual(path.read_bytes(), content)
            with self.assertRaisesRegex(ValueError, 'existing dataset'):
                prepare(*inputs, out)

    def test_tampered_source_and_unmarked_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.fixture(root)
            rows = read_jsonl(inputs[1]); rows[0]['category'] = 'final'
            write_jsonl(inputs[1], rows)
            with self.assertRaisesRegex(ValueError, 'explicitly consumed'):
                prepare(*inputs, root / 'out')
            (inputs[0] / 'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                prepare(*inputs, root / 'out')

    def test_fresh_question_in_conversation_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.fixture(root)
            chat = json.loads(inputs[2].read_text())
            chat['turns'][0]['question'] = 'Untouched conversation'
            write_json(inputs[2], chat)
            with self.assertRaisesRegex(ValueError, 'Fresh question leaked'):
                prepare(*inputs, root / 'out')
            self.assertFalse((root / 'out').exists())
