import numpy as np

from models.federated.actm import budget_matched_selection_baselines


def example_inputs():
    probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.55, 0.45],
            [0.70, 0.30],
            [0.50, 0.50],
            [0.80, 0.20],
            [0.60, 0.40],
            [0.95, 0.05],
            [0.52, 0.48],
            [0.75, 0.25],
            [0.65, 0.35],
        ]
    )
    conflicts = np.asarray([False, False, True, False, False, True, False, False, False, False])
    record_ids = np.asarray([f"R{index:02d}" for index in range(len(probabilities))])
    return probabilities, conflicts, record_ids


def test_all_selection_baselines_use_same_ceil_budget():
    probabilities, conflicts, record_ids = example_inputs()
    masks = budget_matched_selection_baselines(
        probabilities, conflicts, record_ids, prompt_budget=0.30, seed=42
    )
    assert set(masks) == {"Random", "Entropy-only", "Margin-only", "Conflict-only"}
    assert {int(mask.sum()) for mask in masks.values()} == {3}


def test_uncertainty_and_conflict_rankings_match_their_named_signal():
    probabilities, conflicts, record_ids = example_inputs()
    masks = budget_matched_selection_baselines(
        probabilities, conflicts, record_ids, prompt_budget=0.30, seed=42
    )
    # Rows 3, 7 and 1 are closest to equal probability and therefore have the
    # highest entropy and smallest top-two margin.
    assert set(np.flatnonzero(masks["Entropy-only"])) == {1, 3, 7}
    assert set(np.flatnonzero(masks["Margin-only"])) == {1, 3, 7}
    # All conflict-positive rows must be selected before a deterministic filler.
    assert masks["Conflict-only"][2]
    assert masks["Conflict-only"][5]


def test_selection_is_reproducible_and_input_order_invariant():
    probabilities, conflicts, record_ids = example_inputs()
    original = budget_matched_selection_baselines(
        probabilities, conflicts, record_ids, prompt_budget=0.30, seed=52
    )
    permutation = np.asarray([4, 8, 1, 9, 0, 6, 2, 7, 5, 3])
    permuted = budget_matched_selection_baselines(
        probabilities[permutation],
        conflicts[permutation],
        record_ids[permutation],
        prompt_budget=0.30,
        seed=52,
    )
    for strategy, mask in original.items():
        original_ids = set(record_ids[mask])
        permuted_ids = set(record_ids[permutation][permuted[strategy]])
        assert original_ids == permuted_ids
