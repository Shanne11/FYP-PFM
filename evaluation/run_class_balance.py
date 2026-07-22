"""Compare standard and training-only class-weighted federated learning."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.analyze_federated_classes import class_rows
from train_proposed import main as run_proposed
from utils.federated_baseline import run_federated_baseline
from utils.metrics import METRIC_COLUMNS


DEFAULT_SEEDS = [42, 52, 62]
DEFAULT_METHODS = ["fedavg", "fedprox", "proposed"]
NUMERIC_METRICS = [column for column in METRIC_COLUMNS if column != "calibration_available"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--reference-root", default="outputs/repeated")
    parser.add_argument("--output", default="outputs/class_balance")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--max-clients", type=int, default=None,
                        help="Smoke tests only; omit for reportable experiments.")
    return parser.parse_args()


def proposed_config(output, seed, rounds, local_epochs, max_clients):
    return SimpleNamespace(
        dataset="dataset/clean_budgetwise.csv", split_manifest="data/experiment_split.json",
        output=str(output), rounds=rounds, local_epochs=local_epochs, learning_rate=0.001,
        entropy_threshold=0.65, margin_threshold=0.15, prompt_budget=0.30,
        min_notes=1, max_clients=max_clients, seed=seed, note_strategy="selective",
        fusion_mode="semantic_anchor", utility_weights=[0.5, 0.3, 0.2],
        disable_utility_weighting=False, class_weighted_loss=True,
    )


def run_weighted(method, output, seed, rounds, local_epochs, max_clients):
    if method == "fedavg":
        return run_federated_baseline(
            "FedAvg", output, rounds=rounds, local_epochs=local_epochs, seed=seed,
            max_clients=max_clients, class_weighted_loss=True,
        )
    if method == "fedprox":
        return run_federated_baseline(
            "FedProx", output, mu=0.01, rounds=rounds, local_epochs=local_epochs,
            seed=seed, max_clients=max_clients, class_weighted_loss=True,
        )
    if method == "proposed":
        run_proposed(proposed_config(output, seed, rounds, local_epochs, max_clients))
        return pd.read_csv(Path(output) / "overall_metrics.csv").iloc[0].to_dict()
    raise ValueError(f"Unknown method: {method}")


def prediction_diagnostics(path, method, seed):
    frame = pd.read_csv(path)
    rows = pd.DataFrame(class_rows(frame, method, seed))
    return {
        "zero_recall_classes": int(rows["zero_recall"].sum()),
        "never_predicted_classes": int(rows["never_predicted"].sum()),
        "active_prediction_classes": int((~rows["never_predicted"]).sum()),
    }


def summarize(runs):
    rows = []
    for (method, loss), group in runs.groupby(["method", "loss"], sort=False):
        row = {"method": method, "loss": loss, "runs": len(group)}
        for metric in NUMERIC_METRICS + [
            "zero_recall_classes", "never_predicted_classes", "active_prediction_classes",
        ]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def deltas(summary):
    rows = []
    indexed = summary.set_index(["method", "loss"])
    for method in summary["method"].unique():
        standard = indexed.loc[(method, "standard")]
        weighted = indexed.loc[(method, "class_weighted")]
        row = {"method": method}
        for metric in NUMERIC_METRICS + [
            "zero_recall_classes", "never_predicted_classes", "active_prediction_classes",
        ]:
            row[f"{metric}_delta"] = weighted[f"{metric}_mean"] - standard[f"{metric}_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def main(config):
    root = Path(config.output); root.mkdir(parents=True, exist_ok=True)
    reference = Path(config.reference_root); rows = []
    for seed in config.seeds:
        for method in config.methods:
            standard = reference / f"seed_{seed}" / method
            if not (standard / "overall_metrics.csv").exists() or not (standard / "predictions.csv").exists():
                raise FileNotFoundError(f"Missing standard reference output in {standard}")
            standard_metrics = pd.read_csv(standard / "overall_metrics.csv").iloc[0].to_dict()
            rows.append({
                "method": method, "seed": seed, "loss": "standard",
                **{metric: standard_metrics[metric] for metric in METRIC_COLUMNS},
                **prediction_diagnostics(standard / "predictions.csv", method, seed),
            })
            weighted = root / f"seed_{seed}" / method
            print(f"\nRunning class-weighted {method} with seed {seed}")
            weighted_metrics = run_weighted(
                method, weighted, seed, config.rounds, config.local_epochs, config.max_clients
            )
            rows.append({
                "method": method, "seed": seed, "loss": "class_weighted",
                **{metric: weighted_metrics[metric] for metric in METRIC_COLUMNS},
                **prediction_diagnostics(weighted / "predictions.csv", method, seed),
            })
    runs = pd.DataFrame(rows); summary = summarize(runs); change = deltas(summary)
    runs.to_csv(root / "class_balance_runs.csv", index=False)
    summary.to_csv(root / "class_balance_summary.csv", index=False)
    change.to_csv(root / "class_balance_deltas.csv", index=False)

    plot = summary.pivot(index="method", columns="loss", values="macro_f1_mean")
    plot.plot(kind="bar", yerr=summary.pivot(index="method", columns="loss", values="macro_f1_std"),
              figsize=(8, 5), capsize=4)
    plt.ylabel("Macro F1 mean +/- sample SD"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(root / "class_balance_macro_f1.png", dpi=160); plt.close()
    print("\nClass-balance summary\n", summary.to_string(index=False))
    print("\nClass-weighted minus standard\n", change.to_string(index=False))


if __name__ == "__main__":
    main(arguments())
