"""Average two compatible LoRA updates exactly using a doubled-rank adapter."""
import argparse
import json
from pathlib import Path
import shutil

from beerllm.common import write_json
from beerllm.recipe_focus import sha


def average(first, second, profile, output):
    import torch
    from safetensors.torch import load_file, save_file
    if output.exists():
        raise ValueError('Preserve existing outputs')
    configs = [json.loads((p / 'adapter_config.json').read_text()) for p in [first, second]]
    for config in configs:
        config['target_modules'] = sorted(config['target_modules'])
    if configs[0] != configs[1] or configs[0].get('rank_pattern') or configs[0].get('alpha_pattern'):
        raise ValueError('Expected matching uniform-rank adapters')
    config = configs[0]
    profile_config = json.loads((profile / 'adapter_config.json').read_text())
    profile_config['target_modules'] = sorted(profile_config['target_modules'])
    if profile_config != config:
        raise ValueError('Profile adapter config differs')
    if config.get('use_rslora') or config.get('use_dora') or config.get('bias') != 'none':
        raise ValueError('Expected ordinary bias-free LoRA')
    a, b = [load_file(str(p / 'adapter_model.safetensors')) for p in [first, second]]
    if a.keys() != b.keys():
        raise ValueError('Adapter factor keys differ')
    result = {}
    for key in a:
        if a[key].shape != b[key].shape:
            raise ValueError('Factor shapes differ')
        if '.lora_A.' in key:
            if a[key].shape[0] != config['r']:
                raise ValueError('Factor rank differs from config')
            result[key] = torch.cat([a[key], b[key]], dim=0)
        elif '.lora_B.' in key:
            result[key] = torch.cat([a[key] * 0.5, b[key] * 0.5], dim=1)
        else:
            raise ValueError('Unexpected non-LoRA tensor')
    # Doubling alpha and rank retains the original alpha/rank multiplier.
    config['r'] *= 2
    config['lora_alpha'] *= 2
    torch.manual_seed(42)
    for key in a:
        if '.lora_A.' not in key:
            continue
        bk = key.replace('.lora_A.', '.lora_B.')
        vector = torch.randn(a[key].shape[1])
        expected = (a[bk] @ (a[key] @ vector) + b[bk] @ (b[key] @ vector)) * 0.5
        actual = result[bk] @ (result[key] @ vector)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    shutil.copytree(profile, output)
    save_file(result, str(output / 'adapter_model.safetensors'))
    write_json(output / 'adapter_config.json', config)
    write_json(output / 'averaging.json', {
        'method': 'Exact equal mean of each layer LoRA delta; concatenated rank, unchanged alpha/rank.',
        'sources': {str(p): {name: sha(p / name) for name in ['adapter_model.safetensors', 'adapter_config.json']}
                    for p in [first, second]},
        'builder_sha256': sha(Path(__file__)), 'parameters': sum(t.numel() for t in result.values()),
        'weight_sha256': sha(output / 'adapter_model.safetensors'),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['first', 'second', 'profile', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    average(**vars(parser.parse_args()))
