import pandas as pd

from evaluation.run_ablations import VARIANTS, deltas_from_full, summarize


def metric_row(variant, seed, value):
    row = {"variant": variant, "seed": seed, "calibration_available": True}
    for metric in [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1", "ece", "brier_score",
    ]:
        row[metric] = value
    return row


def test_required_ablation_variants_are_explicit():
    assert set(VARIANTS) == {
        "full", "without_actm", "without_notes", "simple_concatenation",
        "without_uncertainty_utility", "without_specificity_utility",
        "without_utility_weighting",
    }


def test_ablation_deltas_use_full_method_as_reference():
    runs = pd.DataFrame([
        metric_row("full", 42, 0.5), metric_row("full", 52, 0.7),
        metric_row("without_notes", 42, 0.4), metric_row("without_notes", 52, 0.4),
    ])
    summary = summarize(runs); deltas = deltas_from_full(summary).set_index("variant")
    assert abs(deltas.loc["full", "accuracy_delta_vs_full"]) < 1e-12
    assert abs(deltas.loc["without_notes", "accuracy_delta_vs_full"] + 0.2) < 1e-12


def test_remaining_utility_weights_are_renormalized():
    no_uncertainty = VARIANTS["without_uncertainty_utility"]["utility_weights"]
    no_specificity = VARIANTS["without_specificity_utility"]["utility_weights"]
    assert abs(sum(no_uncertainty) - 1.0) < 1e-12
    assert abs(sum(no_specificity) - 1.0) < 1e-12
