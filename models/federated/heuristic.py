"""Clarification-derived heuristic and bounded signal-adjusted FedAvg.

The transaction score is an author-defined engineering heuristic. Its
uncertainty term compares different model states, so it must not be interpreted
as an isolated causal measure of Smart Note usefulness.
"""

import copy
import re

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from models.federated.actm import predictive_entropy


def clarification_heuristic(
    before_probabilities,
    after_probabilities,
    note_vectors,
    anchor_vectors,
    notes,
    weights=(0.5, 0.3, 0.2),
    token_cap=8,
):
    """Return the heuristic and its three bounded components.

    Components are uncertainty change, lexical-context novelty and bounded
    alphanumeric note length. Empty notes receive no novelty or length signal.
    """
    before = np.asarray(before_probabilities, dtype=float)
    after = np.asarray(after_probabilities, dtype=float)
    component_weights = np.asarray(weights, dtype=float)
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("before and after probabilities must be aligned 2-D arrays")
    if len(notes) != before.shape[0] or note_vectors.shape[0] != before.shape[0]:
        raise ValueError("probabilities, notes and note vectors must have equal rows")
    if anchor_vectors.shape[0] != before.shape[0]:
        raise ValueError("anchor vectors must align with note vectors")
    if component_weights.shape != (3,) or np.any(component_weights < 0):
        raise ValueError("heuristic weights must contain three non-negative values")
    if not np.isclose(component_weights.sum(), 1.0):
        raise ValueError("heuristic weights must sum to 1.0")
    if token_cap < 1:
        raise ValueError("token_cap must be positive")

    before_h = predictive_entropy(before)
    after_h = predictive_entropy(after)
    max_h = np.log(max(before.shape[1], 2))
    uncertainty_change = np.clip((before_h - after_h) / max_h, 0.0, 1.0)

    similarity = np.asarray([
        cosine_similarity(note_vectors[i], anchor_vectors[i])[0, 0]
        if note_vectors[i].nnz and anchor_vectors[i].nnz else 0.0
        for i in range(note_vectors.shape[0])
    ])
    lexical_novelty = np.clip(1.0 - similarity, 0.0, 1.0)
    token_counts = np.asarray([
        len(re.findall(r"[A-Za-z0-9]+", str(note))) for note in notes
    ])
    note_length = np.clip(token_counts / float(token_cap), 0.0, 1.0)
    nonempty = token_counts > 0
    lexical_novelty = lexical_novelty * nonempty
    score = np.clip(
        component_weights[0] * uncertainty_change
        + component_weights[1] * lexical_novelty
        + component_weights[2] * note_length,
        0.0,
        1.0,
    )
    return score, uncertainty_change, lexical_novelty, note_length, nonempty


def _valid_client_result(result):
    count = result.get("sample_count")
    if not isinstance(count, (int, np.integer)) or count < 1:
        return False, "invalid_sample_count"
    weights = result.get("weights")
    if not isinstance(weights, dict) or not weights:
        return False, "missing_parameters"
    if any(not torch.isfinite(value).all().item() for value in weights.values()):
        return False, "non_finite_parameters"
    local_loss = result.get("local_loss")
    if local_loss is not None and not np.isfinite(local_loss):
        return False, "non_finite_local_loss"
    return True, ""


def bounded_signal_adjusted_average(
    client_results,
    multiplier_bounds=(0.75, 1.25),
    min_notes=1,
    neutral_reference=0.5,
    gamma=0.5,
):
    """Aggregate valid updates using sample count and a bounded signal.

    Invalid updates are excluded before normalisation. Valid clients with
    missing, non-finite or insufficient heuristic evidence receive the neutral
    multiplier 1.0.
    """
    if not client_results:
        raise ValueError("At least one client result is required")
    low, high = map(float, multiplier_bounds)
    if not (0 < low <= 1.0 <= high):
        raise ValueError("multiplier bounds must be positive and contain 1.0")
    if min_notes < 1 or gamma < 0 or not np.isfinite(neutral_reference):
        raise ValueError("invalid signal configuration")

    valid_results = []
    excluded_rows = []
    for result in client_results:
        valid, reason = _valid_client_result(result)
        if valid:
            valid_results.append(result)
        else:
            excluded_rows.append({
                "client_id": result.get("client_id"),
                "included": False,
                "exclusion_reason": reason,
            })
    if not valid_results:
        return None, excluded_rows

    counts = np.asarray([result["sample_count"] for result in valid_results], dtype=float)
    base = counts / counts.sum()
    multipliers = []
    fallback_reasons = []
    for result in valid_results:
        heuristic = result.get("mean_clarification_heuristic")
        if heuristic is None:
            heuristic = result.get("mean_note_utility")  # Legacy input alias.
        note_count = int(result.get("note_count", 0))
        if note_count < min_notes:
            multiplier, fallback_reason = 1.0, "insufficient_notes"
        elif heuristic is None or not np.isfinite(heuristic):
            multiplier, fallback_reason = 1.0, "invalid_heuristic"
        else:
            multiplier = float(np.clip(
                1.0 + gamma * (float(heuristic) - neutral_reference), low, high
            ))
            fallback_reason = ""
        multipliers.append(multiplier)
        fallback_reasons.append(fallback_reason)

    multipliers = np.asarray(multipliers)
    unnormalised = base * multipliers
    final = unnormalised / unnormalised.sum()
    averaged = copy.deepcopy(valid_results[0]["weights"])
    for key in averaged:
        averaged[key] = averaged[key] * final[0]
        for index in range(1, len(valid_results)):
            averaged[key] += valid_results[index]["weights"][key] * final[index]

    rows = []
    for index, result in enumerate(valid_results):
        heuristic = result.get("mean_clarification_heuristic")
        if heuristic is None:
            heuristic = result.get("mean_note_utility")
        rows.append({
            "client_id": result.get("client_id"),
            "included": True,
            "exclusion_reason": "",
            "sample_count": int(counts[index]),
            "mean_clarification_heuristic": heuristic,
            "note_count": int(result.get("note_count", 0)),
            "base_weight": float(base[index]),
            "signal_multiplier": float(multipliers[index]),
            "final_coefficient": float(final[index]),
            "heuristic_fallback": bool(fallback_reasons[index]),
            "fallback_reason": fallback_reasons[index],
            # Legacy output aliases retained for existing analysis scripts.
            "mean_note_utility": heuristic,
            "utility_multiplier": float(multipliers[index]),
            "final_weight": float(final[index]),
            "utility_fallback": bool(fallback_reasons[index]),
        })
    return averaged, rows + excluded_rows
