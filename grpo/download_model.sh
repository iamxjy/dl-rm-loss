#!/bin/bash
set -euo pipefail

echo "Downloading Qwen3-0.6B-Base model to cache..."

# Set HuggingFace cache directory
export HF_HOME="/proj/inf-scaling/iris/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

cd /proj/inf-scaling/iris/dl-rm-loss
source venv/bin/activate

# Use a Python script to download the model
python -c "
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import os

hf_cache_dir = os.getenv('HF_HOME')
print(f'Downloading to {hf_cache_dir}...')

print('Downloading config...')
config = AutoConfig.from_pretrained('Qwen/Qwen3-0.6B-Base', cache_dir=hf_cache_dir)
print('Config downloaded!')

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', cache_dir=hf_cache_dir)
print('Tokenizer downloaded!')

print('Downloading model weights (this will take a while)...')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B-Base',
    cache_dir=hf_cache_dir,
    torch_dtype='auto',
    low_cpu_mem_usage=True
)
print('Model downloaded!')

print('All model files downloaded successfully!')
"

