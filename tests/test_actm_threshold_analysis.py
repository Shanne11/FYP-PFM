import numpy as np

from evaluation.analyze_actm_thresholds import eligibility_metrics


def test_conflict_proxy_can_make_every_record_eligible():
    entropy = np.array([0.1, 0.2, 0.3])
    margin = np.array([0.9, 0.8, 0.7])
    conflicts = np.ones(3, dtype=bool)

    result = eligibility_metrics(
        entropy, margin, conflicts, 0.65, 0.15, True, 0.30
    )

    assert result["eligibility_rate"] == 1.0
    assert result["eligibility_count"] == 3


def test_stricter_thresholds_can_create_selectivity_without_conflict():
    entropy = np.array([0.97, 0.98, 0.99, 0.995])
    margin = np.array([0.010, 0.006, 0.003, 0.001])
    conflicts = np.ones(4, dtype=bool)

    result = eligibility_metrics(
        entropy, margin, conflicts, 0.99, 0.002, False, 0.50
    )

    assert result["eligibility_count"] == 2
    assert result["eligibility_rate"] == 0.5
    assert result["budget_fill_rate"] == 1.0
