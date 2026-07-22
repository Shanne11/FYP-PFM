"""Smart Note utility and bounded utility-weighted FedAvg."""

import copy
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.federated.actm import predictive_entropy


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


def note_utility(
    before_probabilities,
    after_probabilities,
    note_vectors,
    anchor_vectors,
    notes,
    weights=(0.5, 0.3, 0.2),
    token_cap=8,
):
    """Return utility and its three components, each bounded to [0, 1]."""
    before_h = predictive_entropy(before_probabilities)
    after_h = predictive_entropy(after_probabilities)
    max_h = np.log(max(before_probabilities.shape[1], 2))
    reduction = np.clip((before_h - after_h) / max_h, 0.0, 1.0)

    similarity = np.asarray([
        cosine_similarity(note_vectors[i], anchor_vectors[i])[0, 0]
        if note_vectors[i].nnz and anchor_vectors[i].nnz else 0.0
        for i in range(note_vectors.shape[0])
    ])
    specificity = np.clip(1.0 - similarity, 0.0, 1.0)

    informative_counts = np.asarray([
        len(re.findall(r"[A-Za-z0-9]+", str(note))) for note in notes
    ])
    effort = np.clip(informative_counts / float(token_cap), 0.0, 1.0)
    nonempty = informative_counts > 0
    specificity = specificity * nonempty
    utility = np.clip(
        weights[0] * reduction + weights[1] * specificity + weights[2] * effort,
        0.0,
        1.0,
    )
    return utility, reduction, specificity, effort, nonempty


def bounded_fedavg(client_results, multiplier_bounds=(0.75, 1.25), min_notes=1):
    """FedAvg sample weights adjusted by a conservative note-utility multiplier.

    A client with missing/invalid utility or too few notes receives multiplier
    1.0. Final aggregation weights are always normalised to sum to one.
    """
    if not client_results:
        raise ValueError("At least one client result is required")
    counts = np.asarray([result["sample_count"] for result in client_results], dtype=float)
    if np.any(counts < 1) or counts.sum() <= 0:
        raise ValueError("Client sample counts must be positive")
    base = counts / counts.sum()
    multipliers = []
    fallback_reasons = []
    low, high = multiplier_bounds
    for result in client_results:
        utility = result.get("mean_note_utility")
        note_count = int(result.get("note_count", 0))
        if note_count < min_notes:
            multiplier = 1.0
            fallback_reason = "insufficient_notes"
        elif utility is None or not np.isfinite(utility):
            multiplier = 1.0
            fallback_reason = "invalid_utility"
        else:
            multiplier = float(np.clip(low + (high - low) * utility, low, high))
            fallback_reason = ""
        multipliers.append(multiplier)
        fallback_reasons.append(fallback_reason)
    multipliers = np.asarray(multipliers)
    unnormalized = base * multipliers
    final = unnormalized / unnormalized.sum()

    averaged = copy.deepcopy(client_results[0]["weights"])
    for key in averaged:
        averaged[key] = averaged[key] * final[0]
        for index in range(1, len(client_results)):
            averaged[key] += client_results[index]["weights"][key] * final[index]
    rows = [{
        "sample_count": int(counts[i]),
        "mean_note_utility": client_results[i].get("mean_note_utility"),
        "note_count": int(client_results[i].get("note_count", 0)),
        "base_weight": float(base[i]),
        "utility_multiplier": float(multipliers[i]),
        "final_weight": float(final[i]),
        "utility_fallback": bool(fallback_reasons[i]),
        "fallback_reason": fallback_reasons[i],
    } for i in range(len(client_results))]
    return averaged, rows
