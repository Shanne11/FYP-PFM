import numpy as np

from utils.class_balance import inverse_frequency_weights


def test_inverse_frequency_weights_use_training_counts_and_mean_one():
    weights = inverse_frequency_weights(np.array([0, 0, 0, 1]), classes=2).numpy()
    assert weights[1] > weights[0]
    assert abs(weights.mean() - 1.0) < 1e-7
    assert abs(weights[1] / weights[0] - 3.0) < 1e-7
