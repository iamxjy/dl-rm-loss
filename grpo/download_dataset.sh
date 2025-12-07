#!/bin/bash
# Pre-download the dataset to cache before running multi-node training

cd /proj/inf-scaling/iris/dl-rm-loss
source venv/bin/activate

echo "Downloading ultrachat_200k dataset to cache..."
python -c "
import os
from datasets import load_dataset

cache_dir = '/proj/inf-scaling/iris/datasets_cache'
os.makedirs(cache_dir, exist_ok=True)

print(f'Downloading to {cache_dir}...')
dataset = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_gen', cache_dir=cache_dir)
print(f'Downloaded! Size: {len(dataset)}')

print('Creating train/test split...')
dataset = dataset.train_test_split(test_size=1000, seed=42)
print(f'Done! Train: {len(dataset[\"train\"])}, Test: {len(dataset[\"test\"])}')
"

echo "Dataset downloaded successfully!"

