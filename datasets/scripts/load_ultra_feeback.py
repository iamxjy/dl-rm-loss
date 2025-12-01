#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

# Adjust these few knobs as needed.
DATASET_NAME = "trl-lib/ultrafeedback_binarized"
OUTPUT_DIR = Path("ultrafeedback_binarized_clean").expanduser()


def ensure_dataset_dict(dataset):
    return dataset if isinstance(dataset, DatasetDict) else DatasetDict({"train": dataset})


def prepare_dataset(dataset):
    prepared = {}

    for name, split in dataset.items():
        rows = split.to_list()
        
        for row in rows:
            row["flipped_pair"] = False
        
        prepared[name] = Dataset.from_list(rows)
        print(f"{name}: {len(split)} examples.")

    return DatasetDict(prepared)


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
    
    print(f"Saved dataset to {OUTPUT_DIR.resolve()}")


def main():
    dataset = ensure_dataset_dict(load_dataset(DATASET_NAME))
    prepared_dataset = prepare_dataset(dataset)
    save_dataset(prepared_dataset)


if __name__ == "__main__":
    main()

