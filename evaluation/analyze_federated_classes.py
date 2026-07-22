"""Explain federated Macro F1 using per-class results across repeated seeds."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


DEFAULT_SEEDS = [42, 52, 62]
DEFAULT_METHODS = ["fedavg", "fedprox", "proposed"]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/repeated")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--output", default="outputs/class_analysis")
    return parser.parse_args()


def label_columns(frame):
    if {"actual_label", "predicted_label"}.issubset(frame.columns):
        return frame["actual_label"].astype(str), frame["predicted_label"].astype(str)
    return frame["actual"].astype(str), frame["predicted"].astype(str)


def class_rows(frame, method, seed):
    actual, predicted = label_columns(frame)
    labels = sorted(set(actual) | set(predicted))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    predicted_counts = predicted.value_counts()
    rows = []
    for index, label in enumerate(labels):
        rows.append({
            "method": method, "seed": seed, "category": label,
            "support": int(support[index]),
            "support_share": support[index] / len(frame),
            "predicted_count": int(predicted_counts.get(label, 0)),
            "predicted_share": predicted_counts.get(label, 0) / len(frame),
            "precision": precision[index], "recall": recall[index], "f1": f1[index],
            "never_predicted": predicted_counts.get(label, 0) == 0,
            "zero_recall": recall[index] == 0,
        })
    return rows


def confusion_rows(frame, method, seed):
    actual, predicted = label_columns(frame)
    errors = pd.DataFrame({"actual": actual, "predicted": predicted})
    errors = errors[errors["actual"] != errors["predicted"]]
    grouped = errors.groupby(["actual", "predicted"]).size().reset_index(name="count")
    grouped.insert(0, "seed", seed); grouped.insert(0, "method", method)
    return grouped


def summarize_classes(runs):
    rows = []
    for (method, category), group in runs.groupby(["method", "category"], sort=False):
        row = {"method": method, "category": category, "seeds": len(group)}
        for metric in ("support", "support_share", "predicted_count", "predicted_share",
                       "precision", "recall", "f1"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        row["never_predicted_seeds"] = int(group["never_predicted"].sum())
        row["zero_recall_seeds"] = int(group["zero_recall"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_confusions(confusions):
    result = confusions.groupby(["method", "actual", "predicted"], as_index=False)["count"].sum()
    totals = result.groupby("method")["count"].transform("sum")
    result["error_share"] = result["count"] / totals
    return result.sort_values(["method", "count"], ascending=[True, False])


def make_conclusion(summary, confusions):
    lines = ["Federated class-level analysis", ""]
    for method in DEFAULT_METHODS:
        data = summary[summary["method"] == method]
        if data.empty:
            continue
        zero = data.loc[data["zero_recall_seeds"] == data["seeds"], "category"].tolist()
        never = data.loc[data["never_predicted_seeds"] == data["seeds"], "category"].tolist()
        best = data.sort_values("f1_mean", ascending=False).iloc[0]
        worst = data.sort_values("f1_mean").iloc[0]
        top_errors = confusions[confusions["method"] == method].head(3)
        pairs = ", ".join(
            f"{row.actual}->{row.predicted} ({int(row['count'])})" for _, row in top_errors.iterrows()
        )
        lines.extend([
            f"{method}:",
            f"  Best class F1: {best['category']} ({best['f1_mean']:.4f})",
            f"  Worst class F1: {worst['category']} ({worst['f1_mean']:.4f})",
            f"  Zero recall in every seed: {', '.join(zero) if zero else 'none'}",
            f"  Never predicted in every seed: {', '.join(never) if never else 'none'}",
            f"  Largest confusion pairs: {pairs}", "",
        ])
    lines.extend([
        "Low Macro F1 is caused by highly uneven category performance: models concentrate",
        "predictions in a small number of categories while several minority classes receive",
        "zero or near-zero recall. Accuracy and Weighted F1 therefore overstate performance",
        "relative to equal-weighted Macro F1.",
    ])
    return "\n".join(lines) + "\n"


def main(config):
    root = Path(config.root); output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    class_data, confusion_data = [], []
    for seed in config.seeds:
        for method in config.methods:
            path = root / f"seed_{seed}" / method / "predictions.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; rerun the repeated-seed experiment first")
            frame = pd.read_csv(path)
            class_data.extend(class_rows(frame, method, seed))
            confusion_data.append(confusion_rows(frame, method, seed))
    runs = pd.DataFrame(class_data)
    confusions = summarize_confusions(pd.concat(confusion_data, ignore_index=True))
    summary = summarize_classes(runs)
    runs.to_csv(output / "class_runs.csv", index=False)
    summary.to_csv(output / "class_summary.csv", index=False)
    confusions.to_csv(output / "confusion_pairs.csv", index=False)
    conclusion = make_conclusion(summary, confusions)
    (output / "class_analysis_conclusion.txt").write_text(conclusion, encoding="utf-8")

    pivot = summary.pivot(index="category", columns="method", values="f1_mean")
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Mean class F1 across seeds"); plt.xticks(rotation=35, ha="right")
    plt.tight_layout(); plt.savefig(output / "class_f1_comparison.png", dpi=160); plt.close()
    print(summary.to_string(index=False)); print("\n" + conclusion)


if __name__ == "__main__":
    main(arguments())
