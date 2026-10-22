import numpy as np

from models.federated.actm import matched_random_selection
from utils.experiment_data import CATEGORY_ORDER
from utils.metrics import classification_summary


def test_matched_random_selection_is_seeded_budget_matched_and_order_invariant():
    record_ids = np.array(["r3", "r1", "r4", "r2", "r5"])
    first = matched_random_selection(record_ids, 0.30, 42)
    second = matched_random_selection(record_ids, 0.30, 42)
    assert np.array_equal(first, second)
    assert first.sum() == 2

    permutation = np.array([3, 0, 4, 1, 2])
    permuted = matched_random_selection(record_ids[permutation], 0.30, 42)
    assert set(record_ids[first]) == set(record_ids[permutation][permuted])


def test_unknown_prediction_does_not_become_a_fourteenth_macro_f1_class():
    actual = np.array([CATEGORY_ORDER[0], CATEGORY_ORDER[1]])
    predicted = np.array([CATEGORY_ORDER[0], "Unknown"])
    fixed = classification_summary(actual, predicted, class_labels=CATEGORY_ORDER)
    union = classification_summary(actual, predicted)
    assert fixed["macro_f1"] != union["macro_f1"]
    assert len(CATEGORY_ORDER) == 13
