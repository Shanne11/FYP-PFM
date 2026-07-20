import pandas as pd

from evaluation.run_repeated_seeds import summarize_runs


def test_repeated_summary_uses_mean_and_sample_standard_deviation():
    runs = pd.DataFrame({
        "method": ["fedavg", "fedavg", "fedavg"],
        "seed": [42, 52, 62],
        "accuracy": [0.1, 0.2, 0.3],
        "macro_precision": [0.1, 0.2, 0.3],
        "macro_recall": [0.1, 0.2, 0.3],
        "macro_f1": [0.1, 0.2, 0.3],
        "weighted_precision": [0.1, 0.2, 0.3],
        "weighted_recall": [0.1, 0.2, 0.3],
        "weighted_f1": [0.1, 0.2, 0.3],
        "ece": [0.3, 0.2, 0.1],
        "brier_score": [0.9, 0.8, 0.7],
    })
    summary = summarize_runs(runs).iloc[0]
    assert summary["seeds"] == "42,52,62"
    assert summary["runs"] == 3
    assert abs(summary["accuracy_mean"] - 0.2) < 1e-12
    assert abs(summary["accuracy_std"] - 0.1) < 1e-12


def test_single_smoke_run_has_zero_reported_standard_deviation():
    row = {"method": "proposed", "seed": 42}
    row.update({metric: 0.5 for metric in [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1", "ece", "brier_score",
    ]})
    summary = summarize_runs(pd.DataFrame([row])).iloc[0]
    assert summary["runs"] == 1
    assert summary["macro_f1_std"] == 0.0
