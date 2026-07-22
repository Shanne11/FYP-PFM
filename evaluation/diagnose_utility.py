"""Diagnose whether Smart Note utility materially changes client aggregation.

By default this reads the three full-method ablation runs. It does not retrain
models, so it can be used to audit already-completed experiments.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUNS = [
    "outputs/ablations/seed_42/full",
    "outputs/ablations/seed_52/full",
    "outputs/ablations/seed_62/full",
]
COMPONENTS = [
    "uncertainty_reduction", "semantic_specificity", "bounded_effort", "utility",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                        help="Proposed-run directories containing aggregation and utility CSVs.")
    parser.add_argument("--output", default="outputs/utility_diagnostics")
    parser.add_argument("--near-identical-tolerance", type=float, default=0.01,
                        help="Absolute distance from the median multiplier.")
    return parser.parse_args()


def describe(values, prefix):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {f"{prefix}_{name}": np.nan for name in
                ("count", "mean", "std", "min", "p25", "median", "p75", "max")}
    return {
        f"{prefix}_count": int(clean.size),
        f"{prefix}_mean": clean.mean(),
        f"{prefix}_std": clean.std(ddof=1) if clean.size > 1 else 0.0,
        f"{prefix}_min": clean.min(), f"{prefix}_p25": clean.quantile(0.25),
        f"{prefix}_median": clean.median(), f"{prefix}_p75": clean.quantile(0.75),
        f"{prefix}_max": clean.max(),
    }


def load_run(path):
    run = Path(path)
    weights_path = run / "aggregation_weights.csv"
    utility_path = run / "utility_scores.csv"
    if not weights_path.exists() or not utility_path.exists():
        raise FileNotFoundError(f"Missing aggregation_weights.csv or utility_scores.csv in {run}")
    weights = pd.read_csv(weights_path)
    utility = pd.read_csv(utility_path)
    required_weights = {
        "round", "client_id", "mean_note_utility", "note_count", "base_weight",
        "utility_multiplier", "final_weight",
    }
    required_utility = set(COMPONENTS)
    if missing := required_weights - set(weights.columns):
        raise ValueError(f"{weights_path} is missing columns: {sorted(missing)}")
    if missing := required_utility - set(utility.columns):
        raise ValueError(f"{utility_path} is missing columns: {sorted(missing)}")

    minimum_notes = 1
    info_path = run / "experiment_info.json"
    if info_path.exists():
        minimum_notes = int(json.loads(info_path.read_text(encoding="utf-8")).get(
            "minimum_notes_for_weighting", 1
        ))
    run_label = f"{run.parent.name}/{run.name}"
    weights["run"] = run_label
    utility["run"] = run_label
    inferred_fallback = (
        (pd.to_numeric(weights["note_count"], errors="coerce") < minimum_notes)
        | pd.to_numeric(weights["mean_note_utility"], errors="coerce").isna()
    )
    if "utility_fallback" not in weights:
        weights["utility_fallback"] = inferred_fallback
    else:
        weights["utility_fallback"] = weights["utility_fallback"].astype(str).str.lower().eq("true")
    weights["absolute_weight_change"] = (weights["final_weight"] - weights["base_weight"]).abs()
    weights["relative_weight_change"] = np.where(
        weights["base_weight"] > 0,
        weights["absolute_weight_change"] / weights["base_weight"], np.nan,
    )
    return weights, utility


def run_summary(weights, tolerance):
    median = weights["utility_multiplier"].median()
    near = (weights["utility_multiplier"] - median).abs() <= tolerance
    row = {
        "run": weights["run"].iloc[0], "client_rounds": len(weights),
        "fallback_count": int(weights["utility_fallback"].sum()),
        "fallback_rate": weights["utility_fallback"].mean(),
        "near_identical_tolerance": tolerance,
        "near_identical_count": int(near.sum()), "near_identical_rate": near.mean(),
    }
    for column, prefix in [
        ("mean_note_utility", "client_mean_utility"),
        ("utility_multiplier", "multiplier"),
        ("absolute_weight_change", "absolute_weight_change"),
        ("relative_weight_change", "relative_weight_change"),
    ]:
        row.update(describe(weights[column], prefix))
    return row


def component_summary(utility):
    rows = []
    for run, group in utility.groupby("run", sort=False):
        for component in COMPONENTS:
            rows.append({"run": run, "component": component,
                         **describe(group[component], "value")})
    for component in COMPONENTS:
        rows.append({"run": "ALL", "component": component,
                     **describe(utility[component], "value")})
    return pd.DataFrame(rows)


def write_conclusion(weights, summary, output):
    all_row = summary.loc[summary["run"] == "ALL"].iloc[0]
    unsupported = (
        all_row["near_identical_rate"] >= 0.80
        or all_row["multiplier_std"] < 0.02
        or all_row["relative_weight_change_mean"] < 0.01
    )
    verdict = "UNSUPPORTED/NEGATIVE" if unsupported else "MECHANISM HAS MEASURABLE WEIGHT VARIATION"
    lines = [
        f"Utility-weighting diagnostic verdict: {verdict}",
        "",
        f"Client-round observations: {len(weights)}",
        f"Client mean utility range: {all_row['client_mean_utility_min']:.6f} to {all_row['client_mean_utility_max']:.6f}",
        f"Utility multiplier range: {all_row['multiplier_min']:.6f} to {all_row['multiplier_max']:.6f}",
        f"Multiplier standard deviation: {all_row['multiplier_std']:.6f}",
        f"Near-identical multiplier rate: {all_row['near_identical_rate']:.2%}",
        f"Fallback-to-1.0 rate: {all_row['fallback_rate']:.2%}",
        f"Mean absolute base/final weight change: {all_row['absolute_weight_change_mean']:.8f}",
        f"Mean relative base/final weight change: {all_row['relative_weight_change_mean']:.2%}",
        "",
        "Do not claim a utility-weighting benefit unless a revised, development-selected mechanism",
        "is rerun and improves the frozen evaluation consistently. Otherwise report this negative finding.",
    ]
    (output / "diagnostic_conclusion.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main(config):
    output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    loaded = [load_run(path) for path in config.runs]
    weights = pd.concat([item[0] for item in loaded], ignore_index=True)
    utility = pd.concat([item[1] for item in loaded], ignore_index=True)
    summaries = [run_summary(group, config.near_identical_tolerance)
                 for _, group in weights.groupby("run", sort=False)]
    all_weights = weights.copy(); all_weights["run"] = "ALL"
    summaries.append(run_summary(all_weights, config.near_identical_tolerance))
    summary = pd.DataFrame(summaries)
    components = component_summary(utility)

    weights.to_csv(output / "client_round_diagnostics.csv", index=False)
    summary.to_csv(output / "utility_diagnostic_summary.csv", index=False)
    components.to_csv(output / "utility_component_summary.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(weights["mean_note_utility"].dropna(), bins=25)
    axes[0, 0].set_title("Client mean utility")
    axes[0, 1].hist(weights["utility_multiplier"], bins=25)
    axes[0, 1].set_title("Utility multipliers")
    axes[1, 0].hist(weights["relative_weight_change"].dropna(), bins=25)
    axes[1, 0].set_title("Relative base/final weight change")
    for component in COMPONENTS[:3]:
        axes[1, 1].hist(utility[component].dropna(), bins=25, alpha=0.5, label=component)
    axes[1, 1].set_title("Utility components"); axes[1, 1].legend(fontsize=8)
    figure.tight_layout(); figure.savefig(output / "utility_diagnostics.png", dpi=160); plt.close(figure)

    conclusion = write_conclusion(weights, summary, output)
    print(summary.to_string(index=False))
    print("\n" + components.to_string(index=False))
    print("\n" + conclusion)


if __name__ == "__main__":
    main(arguments())
