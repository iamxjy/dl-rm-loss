#!/usr/bin/env python3
"""
Simple test script to verify dataset loading works
Run with: python test_dataset_load.py
"""
import os
from datasets import load_dataset

# Set cache directory
cache_dir = os.getenv("HF_DATASETS_CACHE", "/proj/inf-scaling/iris/datasets_cache")
os.makedirs(cache_dir, exist_ok=True)

print(f"Loading dataset to {cache_dir}...")
try:
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen", cache_dir=cache_dir)
    print(f"Dataset loaded successfully! Size: {len(dataset)}")
    
    print("Creating train/test split...")
    dataset = dataset.train_test_split(test_size=1000, seed=42)
    print(f"Split complete. Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

