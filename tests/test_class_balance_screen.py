import numpy as np
import torch

from experiments.federated_diagnosis.run_class_balance_screen import validation_statistics
from utils.class_balance import effective_number_weights


def test_effective_number_weights_are_mean_one_and_favour_minority():
    labels = np.array([0] * 10 + [1] * 2)
    weights = effective_number_weights(labels, classes=2, beta=0.9999).numpy()
    assert np.isclose(weights.mean(), 1.0)
    assert weights[1] > weights[0]


def test_validation_statistics_use_standard_cross_entropy():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = np.array([0, 1])
    expected = torch.nn.CrossEntropyLoss()(logits, torch.tensor(targets)).item()
    result = validation_statistics(logits, targets)
    assert np.isclose(result["validation_standard_cross_entropy"], expected)
    assert result["predicted_class_coverage"] == 2
