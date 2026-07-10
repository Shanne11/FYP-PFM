import os

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate(actual, predicted, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average="weighted", zero_division=0)
    recall = recall_score(actual, predicted, average="weighted", zero_division=0)
    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    weighted_f1 = f1_score(actual, predicted, average="weighted", zero_division=0)

    report = classification_report(actual, predicted, zero_division=0)

    print(report)

    with open(os.path.join(output_folder, "metrics.txt"), "w") as f:
        f.write(f"Accuracy      : {accuracy:.4f}\n")
        f.write(f"Precision     : {precision:.4f}\n")
        f.write(f"Recall        : {recall:.4f}\n")
        f.write(f"Macro F1      : {macro_f1:.4f}\n")
        f.write(f"Weighted F1   : {weighted_f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(actual, predicted)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.close()

    print("Results saved to:", output_folder)
    return {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1
}