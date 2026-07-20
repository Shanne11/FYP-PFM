"""Shared leakage-safe runner for FedAvg and FedProx baselines."""

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import MLP
from utils.experiment_data import load_experiment_data, save_split_indices
from utils.metrics import evaluate
from utils.proposed_features import ProposedFeatureBuilder


def _tensor(matrix):
    return torch.as_tensor(matrix.toarray(), dtype=torch.float32)


def _predict_probabilities(model, matrix):
    model.eval()
    with torch.no_grad():
        logits = model(_tensor(matrix))
    return torch.softmax(logits, dim=1).numpy()


def _local_train(model, matrix, labels, global_weights, mu, epochs, learning_rate):
    loader = DataLoader(
        TensorDataset(_tensor(matrix), torch.as_tensor(labels, dtype=torch.long)),
        batch_size=32, shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(); model.train(); losses = []
    for _ in range(epochs):
        for features, target in loader:
            optimizer.zero_grad(); loss = criterion(model(features), target)
            if mu:
                proximal = sum(((parameter - global_weights[name]) ** 2).sum()
                               for name, parameter in model.named_parameters())
                loss = loss + (mu / 2.0) * proximal
            loss.backward(); optimizer.step(); losses.append(loss.item())
    return copy.deepcopy(model.state_dict()), float(np.mean(losses))


def _sample_weighted_average(results):
    counts = np.asarray([result["sample_count"] for result in results], dtype=float)
    weights = counts / counts.sum(); averaged = copy.deepcopy(results[0]["weights"])
    for key in averaged:
        averaged[key] = averaged[key] * weights[0]
        for index in range(1, len(results)):
            averaged[key] += results[index]["weights"][key] * weights[index]
    return averaged


def run_federated_baseline(name, output, mu=0.0, rounds=10, local_epochs=3,
                           learning_rate=0.001, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    train, validation, test, manifest = load_experiment_data()
    save_split_indices(output, {"train": train, "validation": validation, "test": test})
    builder = ProposedFeatureBuilder().fit(train)

    def parts(frame):
        metadata, notes, _, labels = builder.transform_parts(frame)
        return hstack([metadata, notes], format="csr"), labels

    validation_x, validation_y = parts(validation); test_x, test_y = parts(test)
    input_size = validation_x.shape[1]; classes = len(builder.category_encoder.classes_)
    global_model = MLP(input_size, classes); checkpoint = output / "best_global_model.pt"
    best_macro_f1 = -1.0; round_rows = []; client_rows = []
    client_ids = sorted(train["user_id"].fillna("Unknown").astype(str).unique())
    for round_number in range(1, rounds + 1):
        global_weights = {name: parameter.detach().clone() for name, parameter in global_model.named_parameters()}
        results = []
        for client_id in client_ids:
            client_frame = train[train["user_id"].fillna("Unknown").astype(str) == client_id]
            client_x, client_y = parts(client_frame); local_model = copy.deepcopy(global_model)
            weights, loss = _local_train(
                local_model, client_x, client_y, global_weights, mu, local_epochs, learning_rate
            )
            results.append({"weights": weights, "sample_count": len(client_frame)})
            client_rows.append({"round": round_number, "client_id": client_id,
                                "sample_count": len(client_frame), "local_loss": loss})
        global_model.load_state_dict(_sample_weighted_average(results))
        validation_prediction = _predict_probabilities(global_model, validation_x).argmax(axis=1)
        macro_f1 = f1_score(validation_y, validation_prediction, average="macro", zero_division=0)
        accuracy = accuracy_score(validation_y, validation_prediction)
        round_rows.append({"round": round_number, "validation_accuracy": accuracy,
                           "validation_macro_f1": macro_f1, "clients": len(results)})
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1; torch.save(global_model.state_dict(), checkpoint)
        print(f"{name} round {round_number}/{rounds}: validation macro-F1={macro_f1:.4f}")

    global_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    test_probabilities = _predict_probabilities(global_model, test_x)
    prediction = test_probabilities.argmax(axis=1)
    actual_labels = builder.category_encoder.inverse_transform(test_y)
    predicted_labels = builder.category_encoder.inverse_transform(prediction)
    metrics = evaluate(
        actual_labels, predicted_labels, str(output), test_probabilities,
        builder.category_encoder.classes_,
    )
    pd.DataFrame(round_rows).to_csv(output / "round_metrics.csv", index=False)
    pd.DataFrame(client_rows).to_csv(output / "client_metrics.csv", index=False)
    pd.DataFrame({"transaction_id": test["transaction_id"], "actual": actual_labels,
                  "predicted": predicted_labels}).to_csv(output / "predictions.csv", index=False)
    information = {
        "baseline": name, "aggregation": "sample-weighted FedAvg", "fedprox_mu": mu,
        "train": len(train), "validation": len(validation), "test": len(test),
        "classes": classes, "clients": len(client_ids), "rounds": rounds,
        "local_epochs": local_epochs, "seed": seed,
        "split_manifest_version": manifest["version"], "best_validation_macro_f1": best_macro_f1,
        "test_metrics": metrics,
    }
    (output / "experiment_info.json").write_text(json.dumps(information, indent=2), encoding="utf-8")
    rounds_frame = pd.DataFrame(round_rows)
    plt.figure(figsize=(8, 5)); plt.plot(rounds_frame["round"], rounds_frame["validation_macro_f1"], marker="o")
    plt.xlabel("Communication round"); plt.ylabel("Validation macro-F1"); plt.tight_layout()
    plt.savefig(output / "convergence_curve.png", dpi=160); plt.close()
    return metrics
