import numpy as np
import pandas as pd
import torch

from models.federated.actm import ACTM, ACTMConfig, CrossAccountConflictDetector
from models.federated.utility import bounded_fedavg


def test_actm_obeys_prompt_budget_and_records_reasons():
    probabilities = np.array([[0.50, 0.50], [0.99, 0.01], [0.55, 0.45], [0.90, 0.10]])
    result = ACTM(ACTMConfig(0.5, 0.2, 0.5)).decide(
        probabilities, [False, False, True, False]
    )
    assert result["triggered"].sum() <= 2
    assert result.loc[2, "conflict_triggered"]
    assert {"entropy_triggered", "margin_triggered", "conflict_triggered"}.issubset(result)


def test_conflicts_are_learned_from_training_context_only():
    frame = pd.DataFrame({
        "location": ["Shop", "Shop", "Cafe"],
        "payment_mode": ["card", "cash", "card"],
        "category": ["Personal", "Business", "Food"],
    })
    detector = CrossAccountConflictDetector().fit(frame)
    assert detector.transform(pd.DataFrame({"location": ["Shop", "Cafe"]})).tolist() == [True, False]


def test_bounded_fedavg_uses_sample_base_weight_and_fallback():
    clients = [
        {"weights": {"w": torch.tensor([1.0])}, "sample_count": 3,
         "mean_note_utility": 1.0, "note_count": 2},
        {"weights": {"w": torch.tensor([3.0])}, "sample_count": 1,
         "mean_note_utility": None, "note_count": 0},
    ]
    averaged, rows = bounded_fedavg(clients, min_notes=1)
    assert rows[0]["utility_multiplier"] == 1.25
    assert rows[1]["utility_multiplier"] == 1.0
    assert rows[0]["utility_fallback"] is False
    assert rows[1]["utility_fallback"] is True
    assert rows[1]["fallback_reason"] == "insufficient_notes"
    assert abs(sum(row["final_weight"] for row in rows) - 1.0) < 1e-9
    assert 1.0 < averaged["w"].item() < 3.0
