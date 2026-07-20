"""Adaptive Cognitive Trigger Model (ACTM).

ACTM decides which transactions may use a Smart Note.  It does not score
clients and is deliberately independent from federated aggregation.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return -(probabilities * np.log(probabilities)).sum(axis=1)


def top_two_margin(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape[1] < 2:
        return np.ones(probabilities.shape[0])
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    return top_two[:, 1] - top_two[:, 0]


class CrossAccountConflictDetector:
    """Learns context-dependent category conflicts from training rows only.

    BudgetWise has no merchant/account fields, so the proposed experiment uses
    location as the merchant-context proxy and payment_mode as the account or
    channel proxy.  These names are configurable for a future richer dataset.
    """

    def __init__(self, merchant_col="location", account_col="payment_mode"):
        self.merchant_col = merchant_col
        self.account_col = account_col
        self.conflicting_merchants: set[str] = set()

    def fit(self, frame: pd.DataFrame, label_col="category"):
        required = {self.merchant_col, self.account_col, label_col}
        if not required.issubset(frame.columns):
            self.conflicting_merchants = set()
            return self
        work = frame[list(required)].fillna("Unknown").astype(str)
        dominant = (
            work.groupby([self.merchant_col, self.account_col])[label_col]
            .agg(lambda values: values.value_counts().index[0])
            .reset_index()
        )
        counts = dominant.groupby(self.merchant_col)[label_col].nunique()
        self.conflicting_merchants = set(counts[counts > 1].index.astype(str))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.merchant_col not in frame:
            return np.zeros(len(frame), dtype=bool)
        return (
            frame[self.merchant_col].fillna("Unknown").astype(str)
            .isin(self.conflicting_merchants).to_numpy()
        )


@dataclass(frozen=True)
class ACTMConfig:
    entropy_threshold: float = 0.65
    margin_threshold: float = 0.15
    prompt_budget: float = 0.30


class ACTM:
    """Entropy, top-two-margin, and cross-context selective prompting."""

    def __init__(self, config: ACTMConfig | None = None):
        self.config = config or ACTMConfig()

    def decide(
        self,
        probabilities: np.ndarray,
        conflicts: Iterable[bool],
    ) -> pd.DataFrame:
        entropy = predictive_entropy(probabilities)
        # Normalisation makes alpha comparable when the class count changes.
        normalizer = np.log(max(probabilities.shape[1], 2))
        normalized_entropy = entropy / normalizer
        margin = top_two_margin(probabilities)
        conflicts = np.asarray(list(conflicts), dtype=bool)
        entropy_hit = normalized_entropy > self.config.entropy_threshold
        margin_hit = margin < self.config.margin_threshold
        eligible = entropy_hit | margin_hit | conflicts

        budget = self.config.prompt_budget
        limit = len(eligible) if budget >= 1 else int(np.ceil(len(eligible) * budget))
        priority = normalized_entropy + (1.0 - margin) + conflicts.astype(float)
        selected = np.zeros(len(eligible), dtype=bool)
        eligible_indices = np.flatnonzero(eligible)
        if limit > 0 and len(eligible_indices):
            ranked = eligible_indices[np.argsort(-priority[eligible_indices], kind="stable")]
            selected[ranked[:limit]] = True

        return pd.DataFrame({
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "top_two_margin": margin,
            "entropy_triggered": entropy_hit,
            "margin_triggered": margin_hit,
            "conflict_triggered": conflicts,
            "eligible": eligible,
            "triggered": selected,
        })

    # Kept solely so Baseline 1-5 remain byte-for-byte usable with their legacy
    # FederatedServer. The proposed pipeline never calls this client-trust API.
    def compute_score(self, accuracy, loss, data_size):
        inverse_loss = 1.0 / (float(loss) + 1e-8)
        return float(0.5 * accuracy + 0.3 * inverse_loss + 0.2 * np.log1p(data_size))
