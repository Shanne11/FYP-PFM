"""Run and summarize repeated-seed federated experiments."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_proposed import main as run_proposed
from utils.federated_baseline import run_federated_baseline
from utils.metrics import METRIC_COLUMNS


DEFAULT_SEEDS = [42, 52, 62]
DEFAULT_METHODS = ["fedavg", "fedprox", "proposed"]
NUMERIC_METRICS = [column for column in METRIC_COLUMNS if column != "calibration_available"]


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in runs.groupby("method", sort=False):
        row = {"method": method, "seeds": ",".join(map(str, sorted(group["seed"]))),
               "runs": len(group)}
        for metric in NUMERIC_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def proposed_config(output, seed, rounds, local_epochs, max_clients):
    return SimpleNamespace(
        dataset="dataset/clean_budgetwise.csv",
        split_manifest="data/experiment_split.json",
        output=str(output), rounds=rounds, local_epochs=local_epochs,
        learning_rate=0.001, entropy_threshold=0.65, margin_threshold=0.15,
        prompt_budget=0.30, min_notes=1, max_clients=max_clients, seed=seed,
        note_strategy="selective", fusion_mode="semantic_anchor",
        utility_weights=[0.5, 0.3, 0.2], disable_utility_weighting=False,
    )


def run_method(method, output, seed, rounds, local_epochs, max_clients=None):
    if method == "fedavg":
        run_federated_baseline("FedAvg", output, mu=0.0, rounds=rounds,
                               local_epochs=local_epochs, seed=seed, max_clients=max_clients)
    elif method == "fedprox":
        run_federated_baseline("FedProx", output, mu=0.01, rounds=rounds,
                               local_epochs=local_epochs, seed=seed, max_clients=max_clients)
    elif method == "proposed":
        run_proposed(proposed_config(output, seed, rounds, local_epochs, max_clients))
    else:
        raise ValueError(f"Unknown method: {method}")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--output", default="outputs/repeated")
    parser.add_argument("--max-clients", type=int, default=None,
                        help="Smoke tests only; omit for reportable experiments.")
    return parser.parse_args()


def main(config):
    root = Path(config.output); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in config.seeds:
        for method in config.methods:
            output = root / f"seed_{seed}" / method
            print(f"\nRunning {method} with training seed {seed}")
            run_method(method, output, seed, config.rounds, config.local_epochs, config.max_clients)
            raw = pd.read_csv(output / "overall_metrics.csv").iloc[0].to_dict()
            metrics = {column: raw[column] for column in METRIC_COLUMNS}
            rows.append({"method": method, "seed": seed, **metrics})
    runs = pd.DataFrame(rows); summary = summarize_runs(runs)
    runs.to_csv(root / "repeated_runs.csv", index=False)
    summary.to_csv(root / "repeated_summary.csv", index=False)

    for metric in ("accuracy", "macro_f1", "weighted_f1", "ece", "brier_score"):
        means = summary.set_index("method")[f"{metric}_mean"]
        errors = summary.set_index("method")[f"{metric}_std"]
        plt.figure(figsize=(8, 5)); means.plot(kind="bar", yerr=errors, capsize=4)
        plt.ylabel(f"{metric} (mean +/- sample SD)"); plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(root / f"{metric}_mean_std.png", dpi=160); plt.close()
    print("\nRepeated-seed summary\n", summary.to_string(index=False))


if __name__ == "__main__":
    main(arguments())
