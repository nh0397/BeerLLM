#!/usr/bin/env bash
# Run from any directory. All project artifacts stay inside this checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
mkdir -p outputs
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
if [[ ! -f data/punkapi/raw.json ]]; then
  .venv/bin/python scripts/fetch_recipes.py
fi
if [[ ! -f data/training/train.jsonl ]]; then
  .venv/bin/python -m beerllm.dataset
fi
if [[ ! -f outputs/beer-lora/adapter/adapter_config.json ]]; then
  if compgen -G 'outputs/beer-lora/checkpoint-*' > /dev/null; then
    .venv/bin/python -u -m beerllm.train --resume
  else
    .venv/bin/python -u -m beerllm.train
  fi
fi
.venv/bin/python -u -m beerllm.evaluate
printf '\nFinished. Results: outputs/evaluation.json\n'
