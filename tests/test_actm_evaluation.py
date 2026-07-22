import pandas as pd

from evaluation.summarize_actm import summarize


def test_actm_summary_reports_mean_and_sample_standard_deviation():
    rows = []
    for value in (0.2, 0.4, 0.6):
        row = {metric: value for metric in [
            "eligible_rate", "prompts_per_100", "prompt_precision",
            "note_acceptance_rate", "mean_uncertainty_reduction",
            "ambiguous_accuracy", "ambiguous_macro_f1", "ambiguous_weighted_f1",
            "ambiguous_ece", "ambiguous_brier_score",
        ]}
        for metric in ("accuracy", "macro_f1", "weighted_f1", "ece", "brier_score"):
            row[f"ambiguous_{metric}_delta_vs_overall"] = 0.0
        rows.append(row)
    result = summarize(pd.DataFrame(rows)).iloc[0]
    assert abs(result["prompt_precision_mean"] - 0.4) < 1e-12
    assert abs(result["prompt_precision_std"] - 0.2) < 1e-12
