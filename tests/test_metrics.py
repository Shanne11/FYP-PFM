import math

import numpy as np

from utils.metrics import METRIC_COLUMNS, calibration_bins, metric_summary


def test_perfect_probabilities_have_zero_calibration_error_and_brier_score():
    actual = np.array(["A", "B", "A"])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    table, ece, brier = calibration_bins(actual, probabilities, ["A", "B"])
    assert ece == 0.0
    assert brier == 0.0
    assert table["count"].sum() == len(actual)


def test_probability_summary_has_one_stable_schema():
    actual = np.array([0, 1]); predicted = np.array([0, 1])
    summary = metric_summary(actual, predicted, np.array([[0.8, 0.2], [0.1, 0.9]]), [0, 1])
    assert list(summary) == METRIC_COLUMNS
    assert summary["calibration_available"] is True
    assert not math.isnan(summary["ece"])


def test_non_probability_method_is_explicitly_not_calibrated():
    summary = metric_summary(["A", "B"], ["A", "A"])
    assert list(summary) == METRIC_COLUMNS
    assert summary["calibration_available"] is False
    assert math.isnan(summary["ece"])
    assert math.isnan(summary["brier_score"])
