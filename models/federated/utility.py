"""Backward-compatible aliases for the former utility terminology."""

import copy

import numpy as np

from models.federated.heuristic import (
    bounded_signal_adjusted_average,
    clarification_heuristic,
)


def calculate_utilities(scores):
    """Legacy normalisation retained for unchanged baseline scripts."""
    scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    return scores / scores.sum() if scores.sum() else np.ones(len(scores)) / len(scores)


def utility_weighted_average(client_weights, utilities):
    """Legacy helper retained for unchanged baseline scripts."""
    averaged = copy.deepcopy(client_weights[0])
    for key in averaged:
        averaged[key] = averaged[key] * utilities[0]
        for index in range(1, len(client_weights)):
            averaged[key] += client_weights[index][key] * utilities[index]
    return averaged


def note_utility(*args, **kwargs):
    """Deprecated alias for :func:`clarification_heuristic`."""
    return clarification_heuristic(*args, **kwargs)


def bounded_fedavg(client_results, multiplier_bounds=(0.75, 1.25), min_notes=1):
    """Deprecated adapter preserving the former result schema."""
    averaged, rows = bounded_signal_adjusted_average(
        client_results,
        multiplier_bounds=multiplier_bounds,
        min_notes=min_notes,
    )
    if averaged is None:
        raise ValueError("No valid client updates are available")
    return averaged, [row for row in rows if row.get("included", True)]
