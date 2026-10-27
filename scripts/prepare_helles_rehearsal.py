"""Promote consumed development cases to practice; preserve untouched test inputs."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from beerllm.common import read_jsonl, write_json, write_jsonl
from beerllm.recipe_focus import sha


def prepare(source, cases, dialogue, fresh_suite, fresh_dialogue, output):
    manifest = json.loads((source / 'manifest.json').read_text())
    if any(sha(source / name) != digest for name, digest in manifest['files_sha256'].items()):
        raise ValueError('Frozen source dataset changed')
    train = read_jsonl(source / 'train.jsonl')
    template = train[0]
    system = template['messages'][0]
    consumed = read_jsonl(cases)
    for case in consumed:
        if case['category'] != 'training_fit' or case['messages'][0] != system:
            raise ValueError('Only explicitly consumed cases with the same system prompt may be promoted')
        row = copy.deepcopy(template)
        row.update(task='consumed_development_practice', messages=case['messages'])
        train.append(row)
    conversation = json.loads(dialogue.read_text())
    if conversation['system_prompt'] != system['content']:
        raise ValueError('Dialogue system prompt differs')
    history = [system]
    for turn in conversation['turns']:
        history += [{'role': 'user', 'content': turn['question']},
                    {'role': 'assistant', 'content': turn['reference']}]
        row = copy.deepcopy(template)
        row.update(task='consumed_development_practice', messages=copy.deepcopy(history))
        train.append(row)
    train = list({json.dumps(r['messages'][:-1], sort_keys=True): r for r in train}.values())
    fresh = read_jsonl(fresh_suite)
    fresh_chat = json.loads(fresh_dialogue.read_text())
    reserved = {m['content'] for r in fresh for m in r['messages'] if m['role'] == 'user'}
    reserved.update(t['question'] for t in fresh_chat['turns'])
    if any(m['content'] in reserved for r in train for m in r['messages'] if m['role'] == 'user'):
        raise ValueError('Fresh question leaked into training')
    if output.exists():
        raise ValueError('Preserve the existing dataset; use a new directory')
    output.mkdir(parents=True)
    write_jsonl(output / 'train.jsonl', train)
    write_jsonl(output / 'probes.jsonl', consumed)
    for name in ['validation.jsonl', 'recipe.jsonl']:
        (output / name).write_bytes((source / name).read_bytes())
    write_json(output / 'manifest.json', {
        'scope': 'Consumed development cases are now training-fit checks, not held-out evaluation.',
        'expert_reviewed': False, 'train_examples': len(train),
        'tasks': dict(Counter(r['task'] for r in train)),
        'inputs_sha256': {str(p): sha(p) for p in [source / 'manifest.json', cases, dialogue, fresh_suite, fresh_dialogue]},
        'builder_sha256': sha(Path(__file__)),
        'files_sha256': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    })
    return len(train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'cases', 'dialogue', 'fresh-suite', 'fresh-dialogue', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    print(prepare(**vars(parser.parse_args())))
