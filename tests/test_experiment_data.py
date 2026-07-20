import json

import pandas as pd

from utils.experiment_data import (
    CATEGORY_ALIASES, canonicalize_categories, dataset_fingerprint,
    load_experiment_data,
)


EXPECTED_CATEGORIES = {
    "Bonus", "Education", "Entertainment", "Food", "Freelance", "Health",
    "Investment", "Other", "Rent", "Salary", "Savings", "Travel", "Utilities",
}


def test_every_raw_category_maps_to_the_thirteen_canonical_categories():
    raw = pd.read_csv("dataset/clean_budgetwise.csv")
    cleaned = canonicalize_categories(raw)
    assert set(cleaned["category"].unique()) == EXPECTED_CATEGORIES
    assert len(CATEGORY_ALIASES) == len(raw["category"].str.lower().unique())


def test_frozen_split_is_disjoint_complete_and_matches_the_dataset():
    train, validation, test, manifest = load_experiment_data()
    sets = [set(split["source_index"]) for split in (train, validation, test)]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert sum(map(len, sets)) == 11531
    assert [len(train), len(validation), len(test)] == [7494, 1729, 2308]


def test_each_split_contains_every_category():
    train, validation, test, _ = load_experiment_data()
    for split in (train, validation, test):
        assert set(split["category"].unique()) == EXPECTED_CATEGORIES
