"""Training-only class balancing helpers."""

import numpy as np
import torch


def inverse_frequency_weights(labels, classes):
    """Return mean-one inverse-frequency weights from training labels only."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=classes).astype(float)
    if len(counts) != classes or np.any(counts <= 0):
        raise ValueError("Every output class must occur in the training partition")
    weights = labels.size / (classes * counts)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)
