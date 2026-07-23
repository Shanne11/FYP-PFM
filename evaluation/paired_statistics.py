"""Paired three-seed uncertainty analysis for federated comparisons."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


METRICS = [
    "accuracy", "macro_f1", "weighted_f1", "ece", "brier_score",
    "zero_recall_classes", "active_prediction_classes",
]
LOWER_IS_BETTER = {"ece", "brier_score", "zero_recall_classes"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="outputs/class_balance/class_balance_runs.csv")
    parser.add_argument("--output", default="outputs/statistics")
    return parser.parse_args()


def exact_sign_flip_pvalue(differences):
    """Two-sided exact paired randomization p-value under exchangeability."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(differences.mean())
    permuted = [abs(np.mean(differences * np.asarray(signs)))
                for signs in itertools.product([-1, 1], repeat=len(differences))]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def paired_row(comparison, metric, values_a, values_b, label_a, label_b):
    values_a = np.asarray(values_a, dtype=float); values_b = np.asarray(values_b, dtype=float)
    differences = values_a - values_b; count = len(differences)
    mean_difference = differences.mean()
    sd_difference = differences.std(ddof=1) if count > 1 else 0.0
    critical = t.ppf(0.975, df=count - 1) if count > 1 else np.nan
    margin = critical * sd_difference / np.sqrt(count) if count > 1 else np.nan
    effect = mean_difference / sd_difference if sd_difference > 0 else np.nan
    if metric in LOWER_IS_BETTER:
        wins = int((values_a < values_b).sum())
    else:
        wins = int((values_a > values_b).sum())
    ties = int(np.isclose(values_a, values_b).sum())
    return {
        "comparison": comparison, "metric": metric, "label_a": label_a, "label_b": label_b,
        "paired_seeds": count, "mean_a": values_a.mean(), "mean_b": values_b.mean(),
        "mean_difference_a_minus_b": mean_difference,
        "difference_sd": sd_difference, "ci95_lower": mean_difference - margin,
        "ci95_upper": mean_difference + margin, "paired_effect_dz": effect,
        "wins_a": wins, "ties": ties, "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
    }


def paired_values(runs, method_a, loss_a, method_b, loss_b, metric):
    a = runs[(runs["method"] == method_a) & (runs["loss"] == loss_a)].set_index("seed")
    b = runs[(runs["method"] == method_b) & (runs["loss"] == loss_b)].set_index("seed")
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        raise ValueError(f"At least two shared seeds required for {method_a} versus {method_b}")
    return a.loc[common, metric], b.loc[common, metric]


def analyze(runs):
    rows = []
    for method in ("fedavg", "fedprox", "proposed"):
        for metric in METRICS:
            a, b = paired_values(runs, method, "class_weighted", method, "standard", metric)
            rows.append(paired_row(
                f"{method}: class_weighted vs standard", metric, a, b,
                f"{method}_class_weighted", f"{method}_standard",
            ))
    for loss in ("standard", "class_weighted"):
        for baseline in ("fedavg", "fedprox"):
            for metric in METRICS:
                a, b = paired_values(runs, "proposed", loss, baseline, loss, metric)
                rows.append(paired_row(
                    f"{loss}: proposed vs {baseline}", metric, a, b,
                    f"proposed_{loss}", f"{baseline}_{loss}",
                ))
    return pd.DataFrame(rows)


def conclusion(results):
    macro = results[results["metric"] == "macro_f1"]
    lines = [
        "Paired statistical analysis conclusion", "",
        "Only three matched seeds are available. Confidence intervals are wide and the smallest",
        "possible non-zero two-sided exact sign-flip p-value is 0.25. These analyses quantify",
        "uncertainty but cannot establish conventional p < 0.05 statistical significance.", "",
    ]
    for _, row in macro.iterrows():
        lines.append(
            f"{row['comparison']}: difference={row['mean_difference_a_minus_b']:.6f}, "
            f"95% CI [{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}], "
            f"wins={int(row['wins_a'])}/{int(row['paired_seeds'])}, "
            f"exact p={row['exact_sign_flip_p']:.3f}."
        )
    lines.extend([
        "", "Report observed mean differences and seed-level consistency as exploratory evidence.",
        "Do not describe any three-seed comparison as statistically significant.",
    ])
    return "\n".join(lines) + "\n"


def main(config):
    runs = pd.read_csv(config.runs); output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    results = analyze(runs)
    results.to_csv(output / "paired_statistics.csv", index=False)
    macro = results[results["metric"] == "macro_f1"].copy()
    macro.to_csv(output / "macro_f1_paired_statistics.csv", index=False)
    text = conclusion(results)
    (output / "statistical_conclusion.txt").write_text(text, encoding="utf-8")
    print(macro.to_string(index=False)); print("\n" + text)


if __name__ == "__main__":
    main(arguments())
