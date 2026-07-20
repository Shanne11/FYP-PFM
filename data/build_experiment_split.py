"""Build the reviewed, frozen 65/15/20 transaction-ID manifest."""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.experiment_data import canonicalize_categories, dataset_fingerprint


SEED = 42
DATASET = "dataset/clean_budgetwise.csv"
OUTPUT = "data/experiment_split.json"


def allocate(frame, seed=SEED):
    buckets = {"train": [], "validation": [], "test": []}
    for category, group in frame.groupby("category", sort=True):
        ids = group.index.astype(int).tolist()
        random.Random(f"{seed}:{category}").shuffle(ids)
        test_count = round(len(ids) * 0.20)
        validation_count = round(len(ids) * 0.15)
        buckets["test"].extend(ids[:test_count])
        buckets["validation"].extend(ids[test_count:test_count + validation_count])
        buckets["train"].extend(ids[test_count + validation_count:])
    for name in buckets:
        random.Random(f"{seed}:{name}").shuffle(buckets[name])
    return buckets


if __name__ == "__main__":
    data = canonicalize_categories(pd.read_csv(DATASET))
    split = allocate(data)
    manifest = {
        "version": 2,
        "seed": SEED,
        "strategy": "category-stratified 65/15/20 split",
        "dataset_fingerprint": dataset_fingerprint(data),
        "category_mapping": "utils.experiment_data.CATEGORY_ALIASES",
        "counts": {name: len(ids) for name, ids in split.items()},
        **split,
    }
    Path(OUTPUT).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest["counts"])
