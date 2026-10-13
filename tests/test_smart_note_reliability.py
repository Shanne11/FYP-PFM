import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from experiments.smart_note_reliability.rgsf import (
    GateThresholds, fused_probabilities, gate_mask, reliability_features,
    transition_counts,
)
from utils.class_balance import effective_number_weights
from models.federated.actm import ACTM, ACTMConfig
from train_proposed import set_seed
from utils.experiment_data import load_experiment_data
from utils.proposed_features import ProposedFeatureBuilder


def test_effective_number_weights_use_training_counts_and_mean_one():
    weights = effective_number_weights(np.array([0, 0, 0, 1]), 2, beta=0.9).numpy()
    assert weights[1] > weights[0]
    assert weights.mean() == pytest.approx(1.0)
    expected_ratio = (1 - 0.9 ** 3) / (1 - 0.9)
    assert weights[1] / weights[0] == pytest.approx(expected_ratio)


def test_effective_number_weights_reject_invalid_beta_or_missing_class():
    with pytest.raises(ValueError):
        effective_number_weights(np.array([0, 0]), 2)
    with pytest.raises(ValueError):
        effective_number_weights(np.array([0, 1]), 2, beta=1.0)


def test_gate_accepts_and_rejects_without_labels():
    features = pd.DataFrame({
        "prediction_changed": [True, True, False],
        "confidence_delta": [0.08, -0.02, 0.20],
        "margin_delta": [0.06, 0.10, 0.20],
        "entropy_reduction": [0.05, 0.10, 0.20],
        "lexical_novelty": [0.75, 0.75, 0.75],
    })
    threshold = GateThresholds(0.05, 0.05, 0.01, 0.5)
    assert gate_mask(features, [True, True, True], threshold).tolist() == [True, False, False]
    assert "actual" not in features.columns


def test_probability_shapes_and_transition_counts():
    base = np.array([[0.6, 0.4], [0.7, 0.3], [0.4, 0.6]])
    enhanced = np.array([[0.3, 0.7], [0.2, 0.8], [0.2, 0.8]])
    accepted = np.array([True, False, False])
    final = fused_probabilities(base, enhanced, accepted)
    assert final.shape == base.shape
    counts = transition_counts(
        np.array([1, 0, 0]), base, final, np.array([True, True, True]), accepted
    )
    assert counts["wrong_to_correct"] == 1
    assert counts["correct_to_wrong"] == 0
    assert counts["accepted"] == 1
    assert counts["rejected"] == 2


def test_reliability_features_preserve_class_order_and_dimensions():
    base = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    enhanced = np.array([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2]])
    notes = csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    anchors = csr_matrix([[1.0, 0.0], [1.0, 0.0]])
    result = reliability_features(base, enhanced, notes, anchors)
    assert len(result) == 2
    assert result["prediction_changed"].tolist() == [False, True]
    assert np.isfinite(result.select_dtypes("number").to_numpy()).all()


def test_feature_fit_uses_training_partition_and_preserves_label_order():
    train, validation, test, _ = load_experiment_data()
    builder = ProposedFeatureBuilder().fit(train)
    expected = [
        "Bonus", "Education", "Entertainment", "Food", "Freelance", "Health",
        "Investment", "Other", "Rent", "Salary", "Savings", "Travel", "Utilities",
    ]
    assert builder.category_encoder.classes_.tolist() == expected
    validation_parts = builder.transform_parts(validation)
    test_parts = builder.transform_parts(test)
    assert validation_parts[0].shape[1] == test_parts[0].shape[1] == 68
    assert validation_parts[1].shape[1] == test_parts[1].shape[1] == 37


def test_actm_nominal_budget_and_seed_reproducibility():
    probabilities = np.full((11, 13), 1 / 13)
    actm = ACTM(ACTMConfig(prompt_budget=0.30))
    first = actm.decide(probabilities, np.zeros(11, dtype=bool))["triggered"].to_numpy()
    second = actm.decide(probabilities, np.zeros(11, dtype=bool))["triggered"].to_numpy()
    assert first.sum() == 4
    assert np.array_equal(first, second)
    set_seed(42)
    expected = np.random.random(4)
    set_seed(42)
    assert np.array_equal(expected, np.random.random(4))
