"""Export the saved Helles LoRA as a standalone model; never upload or deploy."""
import argparse
import json
from pathlib import Path
import shutil

from beerllm.common import ROOT, GREEDY_DEFAULTS, load_model, write_json
from beerllm.recipe_focus import sha


def prepare(adapter, output):
    adapter, output = adapter.resolve(), output.resolve()
    if output == adapter or adapter in output.parents:
        raise ValueError('Export outside the original adapter directory')
    profile = json.loads((adapter / 'beerllm.json').read_text())
    config = json.loads((adapter / 'adapter_config.json').read_text())
    if profile['model_id'] != 'Qwen/Qwen2.5-1.5B-Instruct':
        raise ValueError('This export is for the saved Qwen Helles model')
    if config['base_model_name_or_path'] != profile['model_id'] or config['revision'] != profile['model_revision']:
        raise ValueError('Adapter and profile identify different base models')
    origin = {'source_files_sha256': {n: sha(adapter / n) for n in
              ['adapter_model.safetensors', 'adapter_config.json', 'beerllm.json']},
              'builder_sha256': sha(Path(__file__)), 'base_revision': profile['model_revision']}
    manifest = output / 'export.json'
    if output.exists():
        if not manifest.exists():
            raise ValueError('Preserve existing output; choose a new --output directory')
        saved = json.loads(manifest.read_text())
        actual = {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                  if p.is_file() and p != manifest}
        if saved['origin'] != origin or saved['files_sha256'] != actual:
            raise ValueError('Export inputs or files changed; preserve them and use a new output')
        return saved
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise ValueError('Allow at least 8 GiB free for the full float32 export')

    import torch
    from huggingface_hub import hf_hub_download
    model, tokenizer = load_model(profile['model_id'], str(adapter), revision=profile['model_revision'])
    model.eval()
    questions = ['Explain the complete Helicious Helles recipe.',
                 'When do the hops go into the boil?',
                 'Which exact water amount and gelatin dose should I measure?']
    batches = [tokenizer(tokenizer.apply_chat_template(
        [{'role': 'system', 'content': profile['system_prompt']}, {'role': 'user', 'content': q}],
        tokenize=False, add_generation_prompt=True), return_tensors='pt') for q in questions]
    with torch.inference_mode():
        reference = [model(**batch).logits[:, -1, :].clone() for batch in batches]
        merged = model.merge_and_unload(safe_merge=True)
        differences = []
        for batch, expected in zip(batches, reference):
            actual = merged(**batch).logits[:, -1, :]
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            if actual.argmax().item() != expected.argmax().item():
                raise ValueError('Merge changed the next token on a comparison prompt')
            differences.append((actual - expected).abs().max().item())
    for name, value in GREEDY_DEFAULTS.items():
        setattr(merged.generation_config, name, value)
    merged.generation_config.pad_token_id = tokenizer.pad_token_id
    merged.generation_config.eos_token_id = tokenizer.eos_token_id
    merged.save_pretrained(output, safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(output)
    write_json(output / 'beerllm.json', profile)
    license_path = hf_hub_download(profile['model_id'], 'LICENSE', revision=profile['model_revision'])
    shutil.copyfile(license_path, output / 'LICENSE')
    (output / 'README.md').write_text('''---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-1.5B-Instruct
---
# BeerLLM: Helicious Helles

Qwen2.5-1.5B-Instruct with the project's trained Helles LoRA merged into the base weights.
The upstream base-model license is included in LICENSE. These are modified model weights.

This experiment covers one recipe from the [AHA source](https://homebrewersassociation.org/homebrew-recipe/helicious-helles/).
Source omissions include exact water quantities, separate pitching conditions, gelatin dose and carbonation instructions.
References and guides are not expert brewer-reviewed. Other recipes and substitutions are unvalidated.

The unmerged CPU adapter passed twelve fresh wording/conversation checks. The export checks
next-token logit agreement on three prompts; hosted GPU answers still need evaluation.
The serving engine does not automatically read beerllm.json: clients must include its system_prompt
as a system message. Keep the original tokenizer/chat template and use greedy generation.
''')
    result = {'origin': origin, 'dtype': 'float32',
              'merge_check': {'prompts': 3, 'maximum_logit_differences': differences,
                              'next_tokens_unchanged': True},
              'files_sha256': {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}}
    write_json(manifest, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, default=ROOT / 'outputs/helles-demo-v2/adapter')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/helles-hf-v1')
    args = parser.parse_args()
    result = prepare(args.adapter, args.output)
    print(f'Ready to upload: {args.output}\nMerge checks: {result["merge_check"]}')
