"""Leakage-safe Reliability-Gated Smart Note Fusion (RGSF)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from models.federated.actm import predictive_entropy, top_two_margin


@dataclass(frozen=True)
class GateThresholds:
    min_confidence_delta: float = 0.0
    min_margin_delta: float = 0.0
    min_entropy_reduction: float = 0.0
    min_lexical_novelty: float = 0.0


def reliability_features(base_probabilities, note_probabilities, notes, anchors):
    """Derive prediction-time features without labels or future information."""
    base = np.asarray(base_probabilities, dtype=float)
    enhanced = np.asarray(note_probabilities, dtype=float)
    if base.ndim != 2 or base.shape != enhanced.shape:
        raise ValueError("Base and note-enhanced probabilities must be aligned")
    if notes.shape[0] != len(base) or anchors.shape[0] != len(base):
        raise ValueError("Lexical vectors must align with probability rows")
    if not np.isfinite(base).all() or not np.isfinite(enhanced).all():
        raise ValueError("Probabilities must be finite")

    base_pred = base.argmax(axis=1)
    note_pred = enhanced.argmax(axis=1)
    normalizer = np.log(max(base.shape[1], 2))
    novelty = np.asarray([
        1.0 - cosine_similarity(notes[index], anchors[index])[0, 0]
        if notes[index].nnz and anchors[index].nnz else 0.0
        for index in range(len(base))
    ])
    return pd.DataFrame({
        "base_confidence": base.max(axis=1),
        "note_confidence": enhanced.max(axis=1),
        "confidence_delta": enhanced.max(axis=1) - base.max(axis=1),
        "base_margin": top_two_margin(base),
        "note_margin": top_two_margin(enhanced),
        "margin_delta": top_two_margin(enhanced) - top_two_margin(base),
        "base_normalized_entropy": predictive_entropy(base) / normalizer,
        "note_normalized_entropy": predictive_entropy(enhanced) / normalizer,
        "entropy_reduction": (
            predictive_entropy(base) - predictive_entropy(enhanced)
        ) / normalizer,
        "prediction_changed": base_pred != note_pred,
        "prediction_agreement": base_pred == note_pred,
        "lexical_novelty": np.clip(novelty, 0.0, 1.0),
    })


def gate_mask(features, eligible_note_mask, thresholds):
    """Accept only changed note predictions meeting every reliability threshold."""
    eligible = np.asarray(eligible_note_mask, dtype=bool)
    if len(features) != len(eligible):
        raise ValueError("Gate features and eligibility mask must align")
    values = asdict(thresholds)
    accepted = (
        eligible
        & features["prediction_changed"].to_numpy(bool)
        & (features["confidence_delta"].to_numpy() >= values["min_confidence_delta"])
        & (features["margin_delta"].to_numpy() >= values["min_margin_delta"])
        & (features["entropy_reduction"].to_numpy() >= values["min_entropy_reduction"])
        & (features["lexical_novelty"].to_numpy() >= values["min_lexical_novelty"])
    )
    return accepted


def fused_probabilities(base, enhanced, accepted):
    base = np.asarray(base, dtype=float)
    enhanced = np.asarray(enhanced, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    if base.shape != enhanced.shape or len(base) != len(accepted):
        raise ValueError("Probability arrays and accepted mask must align")
    result = base.copy()
    result[accepted] = enhanced[accepted]
    return result


def transition_counts(actual, base, final, note_used, accepted=None):
    """Count mutually interpretable transition and confidence outcomes."""
    actual = np.asarray(actual)
    base = np.asarray(base, dtype=float)
    final = np.asarray(final, dtype=float)
    used = np.asarray(note_used, dtype=bool)
    if len(actual) != len(base) or base.shape != final.shape or len(used) != len(actual):
        raise ValueError("Transition inputs must align")
    base_pred, final_pred = base.argmax(1), final.argmax(1)
    before_correct, after_correct = base_pred == actual, final_pred == actual
    changed = base_pred != final_pred
    accepted_mask = used if accepted is None else np.asarray(accepted, dtype=bool)
    if len(accepted_mask) != len(actual):
        raise ValueError("Accepted mask must align")
    accepted_count = int(accepted_mask.sum())
    return {
        "note_used": int(used.sum()),
        "accepted": accepted_count,
        "rejected": int(used.sum() - accepted_count),
        "acceptance_coverage": float(accepted_count / max(used.sum(), 1)),
        "wrong_to_correct": int((used & ~before_correct & after_correct).sum()),
        "correct_to_wrong": int((used & before_correct & ~after_correct).sum()),
        "wrong_to_different_wrong": int((used & ~before_correct & ~after_correct & changed).sum()),
        "unchanged_correct": int((used & before_correct & after_correct & ~changed).sum()),
        "unchanged_incorrect": int((used & ~before_correct & ~after_correct & ~changed).sum()),
        "net_correct_change": int((used & after_correct).sum() - (used & before_correct).sum()),
        "confident_but_wrong": int((used & (final.max(1) > base.max(1)) & ~after_correct).sum()),
        "accepted_accuracy": float(after_correct[accepted_mask].mean()) if accepted_count else np.nan,
    }


def _candidate_thresholds():
    return (
        GateThresholds(*values)
        for values in product(
            (0.0, 0.01, 0.025, 0.05),
            (0.0, 0.01, 0.025, 0.05),
            (0.0, 0.01, 0.025, 0.05),
            (0.0, 0.25, 0.50, 0.75),
        )
    )


def tune_gate(actual, base, enhanced, features, note_used):
    """Select transparent thresholds on validation labels only.

    Labels are used to compare complete candidate policies, never as an input
    to a transaction's gate decision. Macro F1 is primary; accuracy, net
    corrections, fewer confident-wrong cases and lower coverage break ties.
    """
    actual = np.asarray(actual)
    rows = []
    best = None
    for thresholds in _candidate_thresholds():
        accepted = gate_mask(features, note_used, thresholds)
        final = fused_probabilities(base, enhanced, accepted)
        counts = transition_counts(actual, base, final, note_used, accepted)
        predicted = final.argmax(1)
        row = {
            **asdict(thresholds),
            "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
            "accuracy": accuracy_score(actual, predicted),
            **counts,
        }
        rows.append(row)
        key = (
            row["macro_f1"], row["accuracy"], row["net_correct_change"],
            -row["confident_but_wrong"], -row["acceptance_coverage"],
        )
        if best is None or key > best[0]:
            best = (key, thresholds)
    return best[1], pd.DataFrame(rows)
