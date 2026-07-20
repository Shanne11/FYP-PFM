"""Run the full proposed method and component ablations on identical seeds."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_proposed import main as run_proposed
from utils.metrics import METRIC_COLUMNS


DEFAULT_SEEDS = [42, 52, 62]
NUMERIC_METRICS = [column for column in METRIC_COLUMNS if column != "calibration_available"]
VARIANTS = {
    "full": {},
    "without_actm": {"note_strategy": "always"},
    "without_notes": {"note_strategy": "none"},
    "simple_concatenation": {"fusion_mode": "simple_concat"},
    "without_uncertainty_utility": {"utility_weights": [0.0, 0.6, 0.4]},
    "without_specificity_utility": {"utility_weights": [5 / 7, 0.0, 2 / 7]},
    "without_utility_weighting": {"disable_utility_weighting": True},
}


def config_for(output, seed, rounds, local_epochs, max_clients, overrides):
    values = {
        "dataset": "dataset/clean_budgetwise.csv",
        "split_manifest": "data/experiment_split.json",
        "output": str(output), "rounds": rounds, "local_epochs": local_epochs,
        "learning_rate": 0.001, "entropy_threshold": 0.65,
        "margin_threshold": 0.15, "prompt_budget": 0.30, "min_notes": 1,
        "max_clients": max_clients, "seed": seed, "note_strategy": "selective",
        "fusion_mode": "semantic_anchor", "utility_weights": [0.5, 0.3, 0.2],
        "disable_utility_weighting": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def summarize(runs):
    rows = []
    for variant, group in runs.groupby("variant", sort=False):
        row = {"variant": variant, "seeds": ",".join(map(str, sorted(group["seed"]))),
               "runs": len(group)}
        for metric in NUMERIC_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def deltas_from_full(summary):
    if "full" not in set(summary["variant"]):
        raise ValueError("Ablation summary requires the full method")
    full = summary.set_index("variant").loc["full"]
    rows = []
    for _, row in summary.iterrows():
        result = {"variant": row["variant"]}
        for metric in NUMERIC_METRICS:
            result[f"{metric}_delta_vs_full"] = row[f"{metric}_mean"] - full[f"{metric}_mean"]
        rows.append(result)
    return pd.DataFrame(rows)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--max-clients", type=int, default=None,
                        help="Smoke tests only; omit for reportable experiments.")
    parser.add_argument("--output", default="outputs/ablations")
    return parser.parse_args()


def main(config):
    root = Path(config.output); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in config.seeds:
        for variant in config.variants:
            output = root / f"seed_{seed}" / variant
            print(f"\nRunning {variant} with training seed {seed}")
            run_proposed(config_for(
                output, seed, config.rounds, config.local_epochs,
                config.max_clients, VARIANTS[variant],
            ))
            raw = pd.read_csv(output / "overall_metrics.csv").iloc[0].to_dict()
            rows.append({"variant": variant, "seed": seed,
                         **{column: raw[column] for column in METRIC_COLUMNS}})
    runs = pd.DataFrame(rows); summary = summarize(runs)
    deltas = (
        deltas_from_full(summary) if "full" in set(summary["variant"])
        else pd.DataFrame(columns=["variant"] + [f"{metric}_delta_vs_full" for metric in NUMERIC_METRICS])
    )
    runs.to_csv(root / "ablation_runs.csv", index=False)
    summary.to_csv(root / "ablation_summary.csv", index=False)
    deltas.to_csv(root / "ablation_deltas.csv", index=False)

    for metric in ("accuracy", "macro_f1", "weighted_f1", "ece", "brier_score"):
        means = summary.set_index("variant")[f"{metric}_mean"]
        errors = summary.set_index("variant")[f"{metric}_std"]
        plt.figure(figsize=(11, 6)); means.plot(kind="bar", yerr=errors, capsize=4)
        plt.ylabel(f"{metric} (mean +/- sample SD)"); plt.xticks(rotation=30, ha="right")
        plt.tight_layout(); plt.savefig(root / f"{metric}_ablation.png", dpi=160); plt.close()
    print("\nAblation summary\n", summary.to_string(index=False))
    print("\nDeltas versus full method\n", deltas.to_string(index=False))


if __name__ == "__main__":
    main(arguments())
