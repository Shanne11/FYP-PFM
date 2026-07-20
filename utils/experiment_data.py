"""Canonical labels and the frozen split shared by every experiment."""

import hashlib
import json
from pathlib import Path

import pandas as pd


CATEGORY_ALIASES = {
    "bonus": "Bonus",
    "edu": "Education", "education": "Education", "educaton": "Education",
    "entertain": "Entertainment", "entertainment": "Entertainment", "entrtnmnt": "Entertainment",
    "fod": "Food", "food": "Food", "foodd": "Food", "foods": "Food",
    "freelance": "Freelance",
    "health": "Health", "helth": "Health",
    "investment": "Investment",
    "misc": "Other", "other": "Other", "others": "Other",
    "rent": "Rent", "rentt": "Rent", "rnt": "Rent",
    "salary": "Salary",
    "saving": "Savings", "savings": "Savings",
    "traval": "Travel", "travel": "Travel", "travl": "Travel",
    "utilities": "Utilities", "utility": "Utilities", "utilties": "Utilities", "utlities": "Utilities",
}


def canonicalize_categories(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    raw = result["category"].fillna("").astype(str).str.strip()
    normalized = raw.str.lower().map(CATEGORY_ALIASES)
    unknown = sorted(raw[normalized.isna()].unique())
    if unknown:
        raise ValueError(f"Unmapped category labels: {unknown}")
    result["category_raw"] = raw
    result["category"] = normalized
    return result


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    stable = frame.drop(columns=["source_index", "category_raw"], errors="ignore")
    serialized = stable.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_experiment_data(
    dataset_path="dataset/clean_budgetwise.csv",
    manifest_path="data/experiment_split.json",
):
    frame = canonicalize_categories(pd.read_csv(dataset_path).reset_index(names="source_index"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    fingerprint = dataset_fingerprint(frame)
    if manifest["dataset_fingerprint"] != fingerprint:
        raise ValueError("The frozen split does not match this dataset. Rebuild and review the manifest.")
    splits = {}
    all_ids = []
    for name in ("train", "validation", "test"):
        ids = manifest[name]
        invalid = [index for index in ids if not isinstance(index, int) or index < 0 or index >= len(frame)]
        if invalid:
            raise ValueError(f"{name} contains invalid source-row indices: {invalid[:3]}")
        splits[name] = frame.iloc[ids].reset_index(drop=True)
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(frame):
        raise ValueError("Frozen splits must be disjoint and cover the complete dataset")
    return splits["train"], splits["validation"], splits["test"], manifest


def save_split_indices(output, splits):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame[["source_index", "transaction_id"]].to_csv(
            output / f"{name}_record_ids.csv", index=False
        )
