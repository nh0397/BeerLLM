"""Ask a deployed Hugging Face vLLM endpoint using the saved Helles instruction."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]


def endpoint_origin(url):
    parts = urlsplit(url)
    if (parts.scheme != 'https' or not parts.hostname or
            not parts.hostname.endswith('.endpoints.huggingface.cloud') or
            parts.username or parts.password or parts.query or parts.fragment or
            parts.path.rstrip('/') not in ['', '/v1']):
        raise ValueError('Use the HTTPS Endpoint URL from your Hugging Face Overview tab')
    return 'https://' + parts.netloc


def query(endpoint, prompt, profile, token, session):
    origin = endpoint_origin(endpoint)
    if not token:
        raise ValueError('Log in first with .venv/bin/hf auth login')
    headers = {'Authorization': 'Bearer ' + token, 'X-Scale-Up-Timeout': '600'}
    # Discover the engine's served name instead of assuming it equals the repository name.
    response = session.get(origin + '/v1/models', headers=headers, timeout=(15, 650))
    response.raise_for_status()
    models = response.json().get('data', [])
    if not models or not isinstance(models[0].get('id'), str):
        raise ValueError('No model is listed yet; check that the endpoint is Running')
    response = session.post(origin + '/v1/chat/completions', headers=headers,
        json={'model': models[0]['id'], 'messages': [
            {'role': 'system', 'content': profile['system_prompt']},
            {'role': 'user', 'content': prompt}],
            'temperature': 0, 'top_p': 1.0, 'max_tokens': 800,
            'frequency_penalty': 0, 'presence_penalty': 0}, timeout=(15, 650))
    response.raise_for_status()
    body = response.json()
    if body['choices'][0].get('finish_reason') == 'length':
        raise ValueError('The response was cut short; inspect endpoint generation settings')
    return body['choices'][0]['message']['content']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--profile', type=Path, default=ROOT / 'outputs/helles-hf-v1/beerllm.json')
    args = parser.parse_args()
    from huggingface_hub import get_token
    profile = json.loads(args.profile.read_text())
    try:
        with requests.Session() as session:
            print(query(args.endpoint, args.prompt, profile, get_token(), session))
    except (ValueError, requests.RequestException) as error:
        parser.exit(1, str(error) + '\n')
