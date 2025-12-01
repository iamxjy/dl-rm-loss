#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

# Adjust these few knobs as needed.
DATASET_NAME = "trl-lib/ultrafeedback_binarized"
NOISE_PROBABILITY = 0.15  # fraction of pairs to swap
SEED = 42
OUTPUT_DIR = Path("ultrafeedback_binarized_noisy").expanduser()


def ensure_dataset_dict(dataset):
    return dataset if isinstance(dataset, DatasetDict) else DatasetDict({"train": dataset})


def add_label_noise(dataset):
    rng = np.random.default_rng(SEED)
    noisy = {}

    for name, split in dataset.items():
        rows = split.to_list()
        
        if name == "train":
            mask = rng.random(len(split)) < NOISE_PROBABILITY
            for idx, flip in enumerate(mask):
                if flip:
                    rows[idx]["chosen"], rows[idx]["rejected"] = rows[idx]["rejected"], rows[idx]["chosen"]
                    rows[idx]["score_chosen"], rows[idx]["score_rejected"] = rows[idx]["score_rejected"], rows[idx]["score_chosen"]
                rows[idx]["flipped_pair"] = bool(flip)
            print(f"{name}: swapped {mask.sum()} / {len(split)} pairs ({mask.mean():.2%}).")
        else:
            for row in rows:
                row["flipped_pair"] = False
            print(f"{name}: {len(split)} examples (no noise added).")
        
        noisy[name] = Dataset.from_list(rows)

    return DatasetDict(noisy)


def save_dataset(dataset):
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, split in dataset.items():
        output_file = OUTPUT_DIR / f"{name}.jsonl"
        rows = split.to_list()
        with open(output_file, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Saved {name} split ({len(rows)} examples) to {output_file}")
    
    print(f"Saved noisy dataset to {OUTPUT_DIR.resolve()}")


def main():
    dataset = ensure_dataset_dict(load_dataset(DATASET_NAME))
    noisy_dataset = add_label_noise(dataset)
    save_dataset(noisy_dataset)


if __name__ == "__main__":
    main()
