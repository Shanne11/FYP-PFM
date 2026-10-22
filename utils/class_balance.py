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


def effective_number_weights(labels, classes, beta=0.9999):
    """Return mean-one class-balanced weights from effective sample counts.

    Implements the effective-number weighting proposed by Cui et al.:
    ``w_c = (1 - beta) / (1 - beta ** n_c)``.  Counts must come from the
    training partition; callers are responsible for preserving that boundary.
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=classes).astype(float)
    if len(counts) != classes or np.any(counts <= 0):
        raise ValueError("Every output class must occur in the training partition")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)
