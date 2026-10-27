"""Copy the saved Helles adapter, retaining its trained feed-forward LoRA layers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'outputs/helles-guide-scope-v1/adapter')
    parser.add_argument('--output', type=Path, default=root/'outputs/helles-mlp-selected-v1/adapter')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output == source or source in output.parents:
        parser.error('Choose a new output outside the source adapter; existing artifacts are preserved.')
    config = json.loads((source/'adapter_config.json').read_text())
    profile = json.loads((source/'beerllm.json').read_text())
    modules = ['gate_proj', 'up_proj', 'down_proj']
    if (config.get('peft_type') != 'LORA' or config.get('bias') != 'none'
            or config.get('modules_to_save') or config.get('use_dora')
            or not set(modules) <= set(config['target_modules'])
            or config['base_model_name_or_path'] != profile['model_id']
            or config.get('revision') != profile['model_revision']):
        parser.error('Expected a matching plain LoRA adapter/profile with all three feed-forward projections.')
    from safetensors.torch import load_file, save_file
    weights = load_file(str(source/'adapter_model.safetensors'))
    selected = {k: v for k, v in weights.items() if any(f'.{m}.lora_' in k for m in modules)}
    for key, tensor in selected.items():
        if '.lora_A.' in key:
            pair = key.replace('.lora_A.', '.lora_B.')
            if pair not in selected or tensor.shape[0] != config['r'] or selected[pair].shape[1] != config['r']:
                parser.error('LoRA factor pairs or ranks differ.')
    if len(selected) != 168 or sum(t.numel() for t in selected.values()) != 14_106_624:
        parser.error('Expected the pinned Helles model: 28 layers with rank-16 feed-forward adapters.')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source_hashes = {name: sha(source/name) for name in ['adapter_model.safetensors', 'adapter_config.json', 'beerllm.json']}
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    config['target_modules'] = modules
    (output/'adapter_config.json').write_text(json.dumps(config, indent=2)+'\n')
    save_file(selected, str(output/'adapter_model.safetensors'), metadata={'format': 'pt'})
    profile['construction'] = {'method': 'retain trained feed-forward LoRA projections; remove attention adapters',
                               'source': str(source), 'source_files_sha256': source_hashes,
                               'modules': modules, 'retained_parameters': 14_106_624,
                               'weights_sha256': sha(output/'adapter_model.safetensors')}
    (output/'beerllm.json').write_text(json.dumps(profile, indent=2)+'\n')
    if any(sha(source/name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError('Source changed during construction; preserve both artifacts and inspect it.')
    print(json.dumps({'output': str(output), **profile['construction']}, indent=2))


if __name__ == '__main__':
    main()
