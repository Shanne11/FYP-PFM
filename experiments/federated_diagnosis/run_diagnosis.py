"""Validation-only Phase 1/2 diagnostics for Tier A federated underfitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "federated_diagnosis" / "outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mlp import MLP
from utils.experiment_data import load_experiment_data
from utils.proposed_features import ProposedFeatureBuilder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[y_true]
    confidence = probabilities.max(axis=1)
    correct = prediction == y_true
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, prediction, average="weighted", zero_division=0),
        "predicted_class_coverage": int(np.unique(prediction).size),
        "nll": log_loss(y_true, probabilities, labels=np.arange(probabilities.shape[1])),
        "ece_10_bin": ece,
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def predict_mlp(model: nn.Module, matrix) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(matrix.toarray(), dtype=torch.float32)
        return torch.softmax(model(tensor), dim=1).cpu().numpy()


def train_central_mlp(train_x, train_y, validation_x, validation_y, seed: int, epochs: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MLP(train_x.shape[1], int(np.max(train_y)) + 1)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x.toarray(), dtype=torch.float32),
                      torch.as_tensor(train_y, dtype=torch.long)),
        batch_size=32, shuffle=True, generator=generator,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_macro = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, target in loader:
            optimiser.zero_grad()
            loss = criterion(model(features), target)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        probabilities = predict_mlp(model, validation_x)
        row = metrics(validation_y, probabilities)
        row.update({"seed": seed, "epoch": epoch, "training_loss": float(np.mean(losses))})
        history.append(row)
        if row["macro_f1"] > best_macro:
            best_macro = row["macro_f1"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def normalised_entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    raw = -float(np.sum(probabilities * np.log(probabilities)))
    return raw / math.log(len(counts)) if len(counts) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]

    dataset = ROOT / "dataset" / "clean_budgetwise.csv"
    split_manifest = ROOT / "data" / "experiment_split.json"
    train, validation, test, manifest = load_experiment_data(dataset, split_manifest)
    builder = ProposedFeatureBuilder().fit(train)
    train_meta, train_notes, _, train_y = builder.transform_parts(train)
    val_meta, val_notes, _, val_y = builder.transform_parts(validation)
    train_all = hstack([train_meta, train_notes], format="csr")
    val_all = hstack([val_meta, val_notes], format="csr")
    feature_sets = {"metadata_68": (train_meta, val_meta), "all_105": (train_all, val_all)}
    categories = list(builder.category_encoder.classes_)

    contract = {
        "source_commit": git_text("rev-parse", "HEAD"),
        "branch": git_text("branch", "--show-current"),
        "dataset_sha256": sha256(dataset),
        "split_manifest_sha256": sha256(split_manifest),
        "split_manifest_version": manifest.get("version"),
        "records": {"train": len(train), "validation": len(validation), "test": len(test)},
        "clients": int(train["user_id"].nunique()),
        "categories": categories,
        "category_count": len(categories),
        "features": {"metadata": builder.metadata_size, "notes": builder.note_size,
                     "total": builder.metadata_size + builder.note_size},
        "preprocessing_fit_partition": "training only",
        "held_out_used_by_new_diagnostics": False,
        "seeds": seeds,
        "central_mlp_epochs": args.epochs,
    }
    (output / "diagnostic_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    global_counts = train["category"].value_counts().reindex(categories, fill_value=0)
    class_rows = pd.DataFrame({"category": categories, "training_count": global_counts.values})
    class_rows["training_share"] = class_rows.training_count / len(train)
    class_rows.to_csv(output / "training_class_distribution.csv", index=False)

    client_rows = []
    work = train.assign(user_id=train.user_id.fillna("Unknown").astype(str))
    for client_id, frame in work.groupby("user_id"):
        counts = frame.category.value_counts().reindex(categories, fill_value=0).to_numpy()
        client_rows.append({
            "client_id": client_id,
            "transactions": len(frame),
            "represented_classes": int(np.count_nonzero(counts)),
            "missing_classes": int(len(categories) - np.count_nonzero(counts)),
            "dominant_class_share": float(counts.max() / counts.sum()),
            "normalised_label_entropy": normalised_entropy(counts),
            "fedavg_base_weight": float(len(frame) / len(train)),
            **{f"count__{category}": int(count) for category, count in zip(categories, counts)},
        })
    clients = pd.DataFrame(client_rows).sort_values("client_id")
    clients.to_csv(output / "client_distribution.csv", index=False)
    client_summary = {
        "clients": len(clients),
        "transactions_min": int(clients.transactions.min()),
        "transactions_median": float(clients.transactions.median()),
        "transactions_max": int(clients.transactions.max()),
        "represented_classes_min": int(clients.represented_classes.min()),
        "represented_classes_median": float(clients.represented_classes.median()),
        "represented_classes_max": int(clients.represented_classes.max()),
        "single_class_clients": int((clients.represented_classes == 1).sum()),
        "clients_with_at_most_3_classes": int((clients.represented_classes <= 3).sum()),
        "dominant_share_mean": float(clients.dominant_class_share.mean()),
        "normalised_entropy_mean": float(clients.normalised_label_entropy.mean()),
        "all_clients_participate_each_round": True,
    }
    (output / "client_distribution_summary.json").write_text(json.dumps(client_summary, indent=2), encoding="utf-8")

    comparison_rows = []
    epoch_rows = []
    class_count = len(categories)
    majority = int(np.bincount(train_y, minlength=class_count).argmax())
    majority_prob = np.full((len(validation), class_count), 1e-12, dtype=float)
    majority_prob[:, majority] = 1.0 - (class_count - 1) * 1e-12
    comparison_rows.append({"method": "majority_class", "features": "none", "seed": "deterministic",
                            "split": "validation", **metrics(val_y, majority_prob)})

    for feature_name, (training_x, validation_x) in feature_sets.items():
        for class_weight in (None, "balanced"):
            logistic = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=class_weight,
                                          random_state=42).fit(training_x, train_y)
            comparison_rows.append({
                "method": "central_logistic_balanced" if class_weight else "central_logistic",
                "features": feature_name, "seed": "deterministic", "split": "validation",
                **metrics(val_y, logistic.predict_proba(validation_x)),
            })
        for seed in seeds:
            model, history = train_central_mlp(training_x, train_y, validation_x, val_y,
                                               seed=seed, epochs=args.epochs)
            best_epoch = max(history, key=lambda row: row["macro_f1"])["epoch"]
            comparison_rows.append({"method": "central_mlp", "features": feature_name,
                                    "seed": seed, "split": "validation", "selected_epoch": best_epoch,
                                    **metrics(val_y, predict_mlp(model, validation_x))})
            for row in history:
                row.update({"method": "central_mlp", "features": feature_name})
                epoch_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "centralised_validation_comparison.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(output / "centralised_epoch_metrics.csv", index=False)

    historical = pd.read_csv(ROOT / "outputs" / "repeated" / "repeated_summary.csv")
    historical.to_csv(output / "historical_federated_heldout_summary.csv", index=False)
    validation_rows = []
    for seed in seeds:
        for method in ("fedavg", "fedprox", "proposed"):
            path = ROOT / "outputs" / "repeated" / f"seed_{seed}" / method / "round_metrics.csv"
            frame = pd.read_csv(path)
            macro_column = "validation_macro_f1" if "validation_macro_f1" in frame else "macro_f1"
            accuracy_column = "validation_accuracy" if "validation_accuracy" in frame else "accuracy"
            selected = frame.loc[frame[macro_column].idxmax()]
            validation_rows.append({"method": method, "seed": seed,
                                    "selected_round": int(selected["round"]),
                                    "validation_accuracy": float(selected[accuracy_column]),
                                    "validation_macro_f1": float(selected[macro_column])})
    pd.DataFrame(validation_rows).to_csv(output / "historical_federated_validation_checkpoints.csv", index=False)
    print(json.dumps({"contract": contract, "client_summary": client_summary,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
