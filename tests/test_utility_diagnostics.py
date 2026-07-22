import pandas as pd

from evaluation.diagnose_utility import run_summary


def test_summary_reports_fallback_similarity_and_weight_variation():
    frame = pd.DataFrame({
        "run": ["example"] * 3,
        "mean_note_utility": [0.30, 0.31, None],
        "utility_multiplier": [0.90, 0.905, 1.0],
        "base_weight": [0.2, 0.3, 0.5],
        "final_weight": [0.19, 0.30, 0.51],
        "absolute_weight_change": [0.01, 0.0, 0.01],
        "relative_weight_change": [0.05, 0.0, 0.02],
        "utility_fallback": [False, False, True],
    })
    result = run_summary(frame, tolerance=0.01)
    assert result["fallback_count"] == 1
    assert result["fallback_rate"] == 1 / 3
    assert result["multiplier_min"] == 0.90
    assert result["multiplier_max"] == 1.0
    assert result["near_identical_count"] == 2
