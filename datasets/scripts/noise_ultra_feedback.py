#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

DATASET_NAME = "trl-lib/ultrafeedback_binarized"
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("noise", type=int, nargs="?", default=15)
    args = parser.parse_args()

    dataset = load_dataset(DATASET_NAME)
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    rng = np.random.default_rng(SEED)
    noisy = {}
    for name, split in dataset.items():
        rows = split.to_list()
        if name == "train":
            mask = rng.random(len(rows)) < args.noise / 100.0
            for idx, flip in enumerate(mask):
                if flip:
                    rows[idx]["chosen"], rows[idx]["rejected"] = rows[idx]["rejected"], rows[idx]["chosen"]
                    rows[idx]["score_chosen"], rows[idx]["score_rejected"] = rows[idx]["score_rejected"], rows[idx]["score_chosen"]
                rows[idx]["flipped_pair"] = bool(flip)
            print(f"{name}: swapped {mask.sum()} / {len(rows)} pairs ({mask.mean():.2%}).")
        else:
            for row in rows:
                row["flipped_pair"] = False
            print(f"{name}: {len(rows)} examples (no noise).")
        noisy[name] = Dataset.from_list(rows)

    output_dir = Path(f"datasets/ultrafeedback_binarized_noisy{args.noise}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for name, split in noisy.items():
        with open(output_dir / f"{name}.jsonl", "w") as f:
            for row in split.to_list():
                f.write(json.dumps(row) + "\n")
    print(f"Saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
