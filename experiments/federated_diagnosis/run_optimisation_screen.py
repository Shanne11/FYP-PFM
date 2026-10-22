"""Validation-only Adam versus SGD federated optimisation screen.

The experiment uses metadata-only features and never transforms or evaluates the
held-out test partition. Local optimiser state is intentionally recreated for
every client in every communication round to match the existing implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mlp import MLP
from utils.experiment_data import load_experiment_data
from utils.proposed_features import ProposedFeatureBuilder


DEFAULT_OUTPUT = ROOT / "experiments" / "federated_diagnosis" / "outputs" / "optimisation_screen"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_logits(model: nn.Module, matrix) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(matrix.toarray(), dtype=torch.float32))


def sample_weighted_average(results: list[dict]) -> dict[str, torch.Tensor]:
    counts = np.asarray([row["sample_count"] for row in results], dtype=float)
    coefficients = counts / counts.sum()
    averaged = copy.deepcopy(results[0]["weights"])
    for key in averaged:
        averaged[key] = averaged[key] * coefficients[0]
        for index in range(1, len(results)):
            averaged[key] += results[index]["weights"][key] * coefficients[index]
    return averaged


def local_train(
    model: nn.Module,
    tensor_x: torch.Tensor,
    tensor_y: torch.Tensor,
    optimiser_name: str,
    learning_rate: float,
    momentum: float,
    local_epochs: int,
) -> tuple[dict[str, torch.Tensor], float]:
    loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=32, shuffle=True)
    if optimiser_name == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimiser_name == "sgd":
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimiser: {optimiser_name}")
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_sum = 0.0
    observations = 0
    for _ in range(local_epochs):
        for features, target in loader:
            optimiser.zero_grad()
            loss = criterion(model(features), target)
            loss.backward()
            optimiser.step()
            loss_sum += float(loss.item()) * len(target)
            observations += len(target)
    return copy.deepcopy(model.state_dict()), loss_sum / observations


def run_variant(
    *,
    name: str,
    optimiser: str,
    learning_rate: float,
    momentum: float,
    rounds: int,
    seed: int,
    output: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    builder: ProposedFeatureBuilder,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir = output / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_metadata, _, _, train_labels = builder.transform_parts(train)
    validation_metadata, _, _, validation_labels = builder.transform_parts(validation)
    classes = len(builder.category_encoder.classes_)
    if train_metadata.shape[1] != 68 or classes != 13:
        raise ValueError("Tier A metadata-only contract requires 68 inputs and 13 categories")

    client_cache = {}
    client_ids = sorted(train.user_id.fillna("Unknown").astype(str).unique())
    if len(client_ids) != 150:
        raise ValueError(f"Expected 150 fixed clients; received {len(client_ids)}")
    user_ids = train.user_id.fillna("Unknown").astype(str).to_numpy()
    for client_id in client_ids:
        indices = np.flatnonzero(user_ids == client_id)
        client_cache[client_id] = (
            torch.as_tensor(train_metadata[indices].toarray(), dtype=torch.float32),
            torch.as_tensor(train_labels[indices], dtype=torch.long),
        )

    global_model = MLP(68, classes)
    validation_target = torch.as_tensor(validation_labels, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    rows = []
    started = time.perf_counter()
    best_macro = -1.0
    for round_number in range(1, rounds + 1):
        client_results = []
        weighted_training_loss = 0.0
        for client_id in client_ids:
            tensor_x, tensor_y = client_cache[client_id]
            local_model = copy.deepcopy(global_model)
            weights, local_loss = local_train(
                local_model, tensor_x, tensor_y, optimiser, learning_rate, momentum, 3
            )
            sample_count = len(tensor_y)
            weighted_training_loss += local_loss * sample_count
            client_results.append({"weights": weights, "sample_count": sample_count})
        global_model.load_state_dict(sample_weighted_average(client_results))
        logits = predict_logits(global_model, validation_metadata)
        probabilities = torch.softmax(logits, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
        validation_loss = float(criterion(logits, validation_target).item())
        macro_f1 = f1_score(validation_labels, predictions, average="macro", zero_division=0)
        row = {
            "variant": name,
            "optimiser": optimiser,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "seed": seed,
            "round": round_number,
            "training_loss": weighted_training_loss / len(train),
            "validation_loss": validation_loss,
            "validation_accuracy": accuracy_score(validation_labels, predictions),
            "validation_macro_f1": macro_f1,
            "predicted_class_coverage": int(np.unique(predictions).size),
            "single_class_collapse": bool(np.unique(predictions).size == 1),
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        if macro_f1 > best_macro:
            best_macro = macro_f1
            torch.save(global_model.state_dict(), run_dir / "best_validation_model.pt")
        print(
            f"{name} round {round_number:02d}/{rounds}: "
            f"acc={row['validation_accuracy']:.4f} macro_f1={macro_f1:.4f} "
            f"coverage={row['predicted_class_coverage']}"
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / "round_metrics.csv", index=False)
    selected = metrics.loc[metrics.validation_macro_f1.idxmax()]
    final = metrics.iloc[-1]
    summary = {
        "variant": name,
        "optimiser": optimiser,
        "local_optimiser_state": "reset for each client in each round",
        "learning_rate": learning_rate,
        "momentum": momentum,
        "seed": seed,
        "rounds": rounds,
        "local_epochs": 3,
        "batch_size": 32,
        "clients_per_round": 150,
        "features": "68 metadata features; Smart Note block excluded",
        "loss": "standard cross-entropy",
        "selection_partition": "validation",
        "held_out_evaluated": False,
        "best_round": int(selected["round"]),
        "best_validation_accuracy": float(selected["validation_accuracy"]),
        "best_validation_macro_f1": float(selected["validation_macro_f1"]),
        "best_predicted_class_coverage": int(selected["predicted_class_coverage"]),
        "round_25_validation_accuracy": float(final["validation_accuracy"]) if rounds == 25 else None,
        "round_25_validation_macro_f1": float(final["validation_macro_f1"]) if rounds == 25 else None,
        "round_25_predicted_class_coverage": int(final["predicted_class_coverage"]) if rounds == 25 else None,
        "runtime_seconds": float(metrics.elapsed_seconds.iloc[-1]),
    }
    (run_dir / "config_and_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_manifest(output: Path, experiment: dict) -> None:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append({
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    payload = {"experiment": experiment, "artifact_count": len(artifacts), "artifacts": artifacts}
    (output / "artifact_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tuning-rounds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--sgd-learning-rates", default="0.01,0.03,0.1")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.manifest_only:
        experiment = json.loads((output / "stage1_experiment.json").read_text(encoding="utf-8"))
        write_manifest(output, experiment)
        print(f"Refreshed manifest for {output}")
        return

    train, validation, _held_out_unused, manifest = load_experiment_data()
    builder = ProposedFeatureBuilder().fit(train)
    candidates = [float(value) for value in args.sgd_learning_rates.split(",")]
    experiment_started = time.perf_counter()

    tuning = []
    for learning_rate in candidates:
        summary = run_variant(
            name=f"sgd_tuning_lr_{learning_rate:g}", optimiser="sgd",
            learning_rate=learning_rate, momentum=0.9, rounds=args.tuning_rounds,
            seed=args.seed, output=output, train=train, validation=validation, builder=builder,
        )
        tuning.append(summary)
    tuning_frame = pd.DataFrame(tuning).sort_values(
        ["best_validation_macro_f1", "best_validation_accuracy"], ascending=False
    )
    tuning_frame.to_csv(output / "sgd_tuning_summary.csv", index=False)
    selected_lr = float(tuning_frame.iloc[0]["learning_rate"])

    adam = run_variant(
        name="adam_reset_seed_42", optimiser="adam", learning_rate=0.001, momentum=0.0,
        rounds=args.rounds, seed=args.seed, output=output, train=train,
        validation=validation, builder=builder,
    )
    sgd = run_variant(
        name="sgd_reset_seed_42", optimiser="sgd", learning_rate=selected_lr, momentum=0.9,
        rounds=args.rounds, seed=args.seed, output=output, train=train,
        validation=validation, builder=builder,
    )
    comparison = pd.DataFrame([adam, sgd])
    comparison.to_csv(output / "stage1_comparison.csv", index=False)
    experiment = {
        "stage": "one-seed validation screening",
        "seed": args.seed,
        "sgd_candidates": candidates,
        "sgd_selection_rule": "highest best-round validation Macro F1 in the 10-round screen; validation accuracy breaks ties",
        "selected_sgd_learning_rate": selected_lr,
        "sgd_momentum": 0.9,
        "split_manifest_version": manifest.get("version"),
        "held_out_evaluated": False,
        "runtime_seconds": time.perf_counter() - experiment_started,
        "scientific_limitation": "Adam versus SGD changes both optimiser algorithm and state behaviour.",
    }
    (output / "stage1_experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    write_manifest(output, experiment)
    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()
