
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.experiment_data import (
    canonicalize_categories,
    dataset_fingerprint,
    load_experiment_data,
)

DATA = V2 / "data/clean_budgetwise_v2.csv"
OLD = V2 / "data/experiment_split_v2.json"
NEW = V2 / "data/experiment_split_v2_group_safe.json"

frame = pd.read_csv(DATA)
old = json.loads(OLD.read_text(encoding="utf-8"))

assert len(frame) == 14011
assert dataset_fingerprint(canonicalize_categories(frame)) == old["dataset_fingerprint"]

names = ["train", "validation", "test"]
targets = old["counts"]
fractions = {"train": .65, "validation": .15, "test": .20}
totals = frame["category"].value_counts().to_dict()


def stable_hash(value):
    h = 2166136261
    for byte in value.encode("utf-8"):
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


groups = []
for transaction_id, group in frame.groupby("transaction_id", sort=False):
    groups.append({
        "id": str(transaction_id),
        "indices": group.index.tolist(),
        "categories": group["category"].value_counts().to_dict()
    })

# Allocate repeated IDs together, largest groups first.
groups.sort(key=lambda g: (
    -len(g["indices"]),
    stable_hash("42:" + g["id"]),
    g["id"],
))

splits = {name: [] for name in names}
sizes = Counter()
class_counts = {name: Counter() for name in names}

for group in groups:
    group_size = len(group["indices"])
    choices = []

    for name in names:
        if sizes[name] + group_size > targets[name]:
            continue

        class_fill = sum(
            count * class_counts[name][category]
            / (totals[category] * fractions[name])
            for category, count in group["categories"].items()
        ) / group_size

        score = (
            class_fill
            + 0.35 * sizes[name] / targets[name]
            + 1e-8 * stable_hash(group["id"] + ":" + name) / 2**32
        )
        choices.append((score, name))

    if not choices:
        raise RuntimeError("Unable to place group " + group["id"])

    selected = min(choices)[1]
    splits[selected].extend(group["indices"])
    sizes[selected] += group_size
    class_counts[selected].update(group["categories"])

# Verify all records and IDs are isolated.
assert {name: len(splits[name]) for name in names} == targets

all_rows = [i for name in names for i in splits[name]]
assert len(all_rows) == len(frame)
assert len(set(all_rows)) == len(frame)

id_sets = {
    name: set(frame.iloc[splits[name]]["transaction_id"])
    for name in names
}

for i, a in enumerate(names):
    assert len(class_counts[a]) == 13
    for b in names[i + 1:]:
        assert id_sets[a].isdisjoint(id_sets[b])

manifest = {
    **old,
    "version": 3,
    "strategy": "transaction_id-grouped category-balanced 65/15/20",
    "group_key": "transaction_id",
    "counts": {name: len(splits[name]) for name in names},
    **splits,
}

NEW.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# Verify compatibility with the existing research loader.
load_experiment_data(str(DATA), str(NEW))

print("PASS: Group-safe V2 split created")
print("Train:", len(splits["train"]))
print("Validation:", len(splits["validation"]))
print("Test:", len(splits["test"]))
print("Transaction-ID overlap: 0")
print("Saved:", NEW)
