import numpy as np

from experiments.federated_diagnosis.run_diagnosis import metrics, normalised_entropy


def test_diagnostic_metrics_count_predicted_classes():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
    result = metrics(np.array([0, 1, 1]), probabilities)
    assert result["predicted_class_coverage"] == 2
    assert result["accuracy"] == 2 / 3


def test_normalised_entropy_is_bounded():
    assert normalised_entropy(np.array([10, 0, 0])) == 0.0
    assert np.isclose(normalised_entropy(np.array([5, 5])), 1.0)
