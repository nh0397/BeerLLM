import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import requests

from scripts.prepare_hf_model import prepare
from scripts.query_hf_endpoint import query


class HFHostingTests(unittest.TestCase):
    endpoint = 'https://example.us-east-1.aws.endpoints.huggingface.cloud'

    def session(self, reason='stop'):
        session = Mock()
        session.get.return_value.json.return_value = {'data': [{'id': '/repository'}]}
        session.post.return_value.json.return_value = {
            'choices': [{'finish_reason': reason, 'message': {'content': 'The answer.'}}]}
        return session

    def test_uses_served_model_and_saved_system_instruction(self):
        session = self.session()
        result = query(self.endpoint + '/v1/', 'Explain it.', {'system_prompt': 'Saved instruction.'},
                       'fake-token', session)
        self.assertEqual(result, 'The answer.')
        self.assertEqual(session.get.call_args.args[0], self.endpoint + '/v1/models')
        payload = session.post.call_args.kwargs['json']
        self.assertEqual(payload['model'], '/repository')
        self.assertEqual(payload['messages'], [
            {'role': 'system', 'content': 'Saved instruction.'},
            {'role': 'user', 'content': 'Explain it.'}])
        self.assertEqual(payload['temperature'], 0)
        self.assertNotIn('fake-token', json.dumps(payload))
        self.assertEqual(session.post.call_args.kwargs['headers']['Authorization'], 'Bearer fake-token')

    def test_wrong_host_or_url_cannot_receive_token(self):
        session = self.session()
        for endpoint in ['http://example.endpoints.huggingface.cloud',
                         'https://example.endpoints.huggingface.cloud.other.test',
                         self.endpoint + '/v1/chat/completions',
                         self.endpoint + '?token=mistake',
                         'https://user@example.endpoints.huggingface.cloud']:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                query(endpoint, 'Question', {'system_prompt': 'Saved'}, 'fake-token', session)
        session.get.assert_not_called()

    def test_missing_login_does_not_send_a_request(self):
        session = self.session()
        with self.assertRaisesRegex(ValueError, 'Log in first'):
            query(self.endpoint, 'Question', {'system_prompt': 'Saved'}, None, session)
        session.get.assert_not_called()

    def test_failed_model_discovery_does_not_attempt_generation(self):
        session = self.session()
        session.get.return_value.raise_for_status.side_effect = requests.HTTPError('503')
        with self.assertRaises(requests.HTTPError):
            query(self.endpoint, 'Question', {'system_prompt': 'Saved'}, 'fake-token', session)
        session.post.assert_not_called()

    def test_truncated_output_is_not_reported_as_a_complete_answer(self):
        with self.assertRaisesRegex(ValueError, 'cut short'):
            query(self.endpoint, 'Question', {'system_prompt': 'Saved'}, 'fake-token', self.session('length'))

    def test_export_cannot_target_the_original_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'adapter'
            source.mkdir()
            for target in [source, source / 'nested-export']:
                with self.assertRaisesRegex(ValueError, 'outside the original'):
                    prepare(source, target)
            self.assertEqual(list(source.iterdir()), [])

    def test_export_preserves_existing_non_export_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source = root / 'adapter';source.mkdir()
            model = 'Qwen/Qwen2.5-1.5B-Instruct';revision = 'a' * 40
            (source / 'beerllm.json').write_text(json.dumps({'model_id': model, 'model_revision': revision}))
            (source / 'adapter_config.json').write_text(json.dumps({
                'base_model_name_or_path': model, 'revision': revision}))
            (source / 'adapter_model.safetensors').write_bytes(b'preserved adapter')
            output = root / 'existing';output.mkdir();marker = output / 'notes.txt';marker.write_text('Keep me')
            with self.assertRaisesRegex(ValueError, 'Preserve existing output'):
                prepare(source, output)
            self.assertEqual(marker.read_text(), 'Keep me')
            self.assertEqual((source / 'adapter_model.safetensors').read_bytes(), b'preserved adapter')
