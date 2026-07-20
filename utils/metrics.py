"""Shared classification and probability-calibration metrics."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)


METRIC_COLUMNS = [
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
    "ece", "brier_score", "calibration_available",
]


def classification_summary(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": precision_score(actual, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(actual, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "weighted_precision": precision_score(actual, predicted, average="weighted", zero_division=0),
        "weighted_recall": recall_score(actual, predicted, average="weighted", zero_division=0),
        "weighted_f1": f1_score(actual, predicted, average="weighted", zero_division=0),
    }


def calibration_bins(actual, probabilities, probability_labels, bins=10):
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(probability_labels)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Probability columns must match probability_labels")
    if len(probabilities) != len(actual):
        raise ValueError("Probability rows must match actual labels")
    lookup = {label: index for index, label in enumerate(labels.tolist())}
    try:
        actual_indices = np.asarray([lookup[label] for label in np.asarray(actual).tolist()])
    except KeyError as error:
        raise ValueError(f"Actual label has no probability column: {error.args[0]}") from error
    confidence = probabilities.max(axis=1)
    predicted_indices = probabilities.argmax(axis=1)
    correct = predicted_indices == actual_indices
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        rows.append({
            "bin_lower": lower, "bin_upper": upper, "count": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
            "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
        })
    one_hot = np.eye(len(labels))[actual_indices]
    ece = sum(
        row["count"] / len(actual) * abs(row["accuracy"] - row["mean_confidence"])
        for row in rows if row["count"]
    )
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return pd.DataFrame(rows), float(ece), brier


def metric_summary(actual, predicted, probabilities=None, probability_labels=None):
    summary = classification_summary(actual, predicted)
    if probabilities is None:
        summary.update({"ece": np.nan, "brier_score": np.nan, "calibration_available": False})
        return summary
    _, ece, brier = calibration_bins(actual, probabilities, probability_labels)
    summary.update({"ece": ece, "brier_score": brier, "calibration_available": True})
    return summary


def evaluate(actual, predicted, output_folder, probabilities=None, probability_labels=None):
    os.makedirs(output_folder, exist_ok=True)
    actual = np.asarray(actual); predicted = np.asarray(predicted)
    summary = metric_summary(actual, predicted, probabilities, probability_labels)
    report = classification_report(actual, predicted, zero_division=0)
    print(report)
    with open(os.path.join(output_folder, "metrics.txt"), "w", encoding="utf-8") as file:
        for key in METRIC_COLUMNS:
            file.write(f"{key}: {summary[key]}\n")
        file.write("\n" + report)

    labels = np.unique(np.concatenate((actual, predicted)))
    matrix = confusion_matrix(actual, predicted, labels=labels)
    plt.figure(figsize=(10, 8)); plt.imshow(matrix); plt.title("Confusion Matrix"); plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90); plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png")); plt.close()

    pd.DataFrame([[summary[column] for column in METRIC_COLUMNS]], columns=METRIC_COLUMNS).to_csv(
        os.path.join(output_folder, "overall_metrics.csv"), index=False
    )
    if probabilities is not None:
        table, _, _ = calibration_bins(actual, probabilities, probability_labels)
        table.to_csv(os.path.join(output_folder, "calibration_metrics.csv"), index=False)
        plotted = table.dropna()
        plt.figure(figsize=(6, 6)); plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.plot(plotted["mean_confidence"], plotted["accuracy"], marker="o")
        plt.xlabel("Mean confidence"); plt.ylabel("Observed accuracy"); plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "reliability_diagram.png"), dpi=160); plt.close()
    return summary
