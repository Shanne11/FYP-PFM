import numpy as np
import pandas as pd
import torch

from models.federated.actm import ACTM, ACTMConfig, CrossAccountConflictDetector
from models.federated.heuristic import bounded_signal_adjusted_average


def test_actm_obeys_prompt_budget_and_records_reasons():
    probabilities = np.array([[0.50, 0.50], [0.99, 0.01], [0.55, 0.45], [0.90, 0.10]])
    result = ACTM(ACTMConfig(0.5, 0.2, 0.5)).decide(
        probabilities, [False, False, True, False]
    )
    assert result["triggered"].sum() <= 2
    assert result.loc[2, "conflict_triggered"]
    assert {"entropy_triggered", "margin_triggered", "conflict_triggered", "priority"}.issubset(result)


def test_conflicts_are_learned_from_training_context_only():
    frame = pd.DataFrame({
        "location": ["Shop", "Shop", "Cafe"],
        "payment_mode": ["card", "cash", "card"],
        "category": ["Personal", "Business", "Food"],
    })
    detector = CrossAccountConflictDetector().fit(frame)
    assert detector.transform(pd.DataFrame({"location": ["Shop", "Cafe"]})).tolist() == [True, False]


def test_bounded_signal_uses_sample_base_weight_and_neutral_fallback():
    clients = [
        {"weights": {"w": torch.tensor([1.0])}, "sample_count": 3,
         "mean_clarification_heuristic": 1.0, "note_count": 2},
        {"weights": {"w": torch.tensor([3.0])}, "sample_count": 1,
         "mean_clarification_heuristic": None, "note_count": 0},
    ]
    averaged, rows = bounded_signal_adjusted_average(clients, min_notes=1)
    assert rows[0]["signal_multiplier"] == 1.25
    assert rows[1]["signal_multiplier"] == 1.0
    assert rows[0]["heuristic_fallback"] is False
    assert rows[1]["heuristic_fallback"] is True
    assert rows[1]["fallback_reason"] == "insufficient_notes"
    assert abs(sum(row["final_coefficient"] for row in rows) - 1.0) < 1e-9
    assert 1.0 < averaged["w"].item() < 3.0


def test_invalid_client_update_is_excluded_before_normalisation():
    clients = [
        {"client_id": "valid", "weights": {"w": torch.tensor([1.0])},
         "sample_count": 2, "mean_clarification_heuristic": .5,
         "note_count": 1, "local_loss": .2},
        {"client_id": "invalid", "weights": {"w": torch.tensor([float("nan")])},
         "sample_count": 2, "mean_clarification_heuristic": .5,
         "note_count": 1, "local_loss": .2},
    ]
    averaged, rows = bounded_signal_adjusted_average(clients)
    assert averaged["w"].item() == 1.0
    assert rows[0]["final_coefficient"] == 1.0
    assert rows[1]["included"] is False
    assert rows[1]["exclusion_reason"] == "non_finite_parameters"
