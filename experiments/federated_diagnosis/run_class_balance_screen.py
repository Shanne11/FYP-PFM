"""Validation-only standard versus effective-number federated loss screen."""

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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mlp import MLP
from utils.class_balance import effective_number_weights
from utils.experiment_data import load_experiment_data
from utils.proposed_features import ProposedFeatureBuilder


DEFAULT_OUTPUT = ROOT / "experiments" / "federated_diagnosis" / "outputs" / "class_balance_screen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(results: list[dict]) -> dict[str, torch.Tensor]:
    counts = np.asarray([row["sample_count"] for row in results], dtype=float)
    coefficients = counts / counts.sum()
    state = copy.deepcopy(results[0]["weights"])
    for key in state:
        state[key] = state[key] * coefficients[0]
        for index in range(1, len(results)):
            state[key] += results[index]["weights"][key] * coefficients[index]
    return state


def validation_statistics(logits: torch.Tensor, targets: np.ndarray) -> dict:
    """Return common unweighted validation measures for either training loss."""
    predictions = logits.argmax(dim=1).numpy()
    standard_ce = float(nn.CrossEntropyLoss()(logits, torch.as_tensor(targets, dtype=torch.long)).item())
    return {
        "validation_standard_cross_entropy": standard_ce,
        "validation_accuracy": accuracy_score(targets, predictions),
        "validation_macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "validation_weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0),
        "predicted_class_coverage": int(np.unique(predictions).size),
        "single_class_collapse": bool(np.unique(predictions).size == 1),
        "predictions": predictions,
    }


def local_train(
    model: nn.Module, features: torch.Tensor, targets: torch.Tensor,
    learning_rate: float, class_weights: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor], float]:
    loader = DataLoader(TensorDataset(features, targets), batch_size=32, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.train()
    loss_sum = 0.0
    observations = 0
    for _ in range(3):
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            loss_sum += float(loss.item()) * len(batch_y)
            observations += len(batch_y)
    return copy.deepcopy(model.state_dict()), loss_sum / observations


def run_variant(
    *, name: str, class_weights: torch.Tensor | None, seed: int, rounds: int,
    output: Path, train: pd.DataFrame, validation: pd.DataFrame,
    builder: ProposedFeatureBuilder, beta: float | None,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir = output / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_x, _, _, train_y = builder.transform_parts(train)
    validation_x, _, _, validation_y = builder.transform_parts(validation)
    categories = list(builder.category_encoder.classes_)
    if train_x.shape[1] != 68 or len(categories) != 13:
        raise ValueError("Expected the fixed 68-feature, 13-category metadata contract")

    client_ids = sorted(train.user_id.fillna("Unknown").astype(str).unique())
    if len(client_ids) != 150:
        raise ValueError(f"Expected 150 clients; received {len(client_ids)}")
    user_ids = train.user_id.fillna("Unknown").astype(str).to_numpy()
    cache = {}
    for client_id in client_ids:
        indices = np.flatnonzero(user_ids == client_id)
        cache[client_id] = (
            torch.as_tensor(train_x[indices].toarray(), dtype=torch.float32),
            torch.as_tensor(train_y[indices], dtype=torch.long),
        )

    model = MLP(68, 13)
    initial_hash = hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
    ).hexdigest()
    validation_tensor = torch.as_tensor(validation_x.toarray(), dtype=torch.float32)
    round_rows = []
    per_class_rows = []
    distribution_rows = []
    started = time.perf_counter()
    best_macro = -1.0
    for round_number in range(1, rounds + 1):
        client_results = []
        weighted_objective_loss = 0.0
        for client_id in client_ids:
            features, targets = cache[client_id]
            local_model = copy.deepcopy(model)
            weights, objective_loss = local_train(
                local_model, features, targets, 0.001, class_weights
            )
            sample_count = len(targets)
            weighted_objective_loss += objective_loss * sample_count
            client_results.append({"weights": weights, "sample_count": sample_count})
        model.load_state_dict(aggregate(client_results))
        model.eval()
        with torch.no_grad():
            logits = model(validation_tensor)
        statistics = validation_statistics(logits, validation_y)
        predictions = statistics.pop("predictions")
        row = {
            "variant": name,
            "seed": seed,
            "round": round_number,
            "training_objective": "effective_number_weighted_cross_entropy" if class_weights is not None else "standard_cross_entropy",
            "training_objective_loss": weighted_objective_loss / len(train),
            **statistics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        round_rows.append(row)

        precision, recall, f1, support = precision_recall_fscore_support(
            validation_y, predictions, labels=np.arange(13), zero_division=0
        )
        prediction_counts = np.bincount(predictions, minlength=13)
        for index, category in enumerate(categories):
            per_class_rows.append({
                "variant": name, "seed": seed, "round": round_number,
                "category_index": index, "category": category,
                "precision": precision[index], "recall": recall[index],
                "f1": f1[index], "support": int(support[index]),
            })
            distribution_rows.append({
                "variant": name, "seed": seed, "round": round_number,
                "category_index": index, "category": category,
                "predicted_count": int(prediction_counts[index]),
                "predicted_share": float(prediction_counts[index] / len(validation_y)),
            })
        if row["validation_macro_f1"] > best_macro:
            best_macro = row["validation_macro_f1"]
            torch.save(model.state_dict(), run_dir / "best_validation_model.pt")
        print(
            f"{name} round {round_number:02d}/{rounds}: "
            f"acc={row['validation_accuracy']:.4f} macro_f1={row['validation_macro_f1']:.4f} "
            f"coverage={row['predicted_class_coverage']}"
        )

    rounds_frame = pd.DataFrame(round_rows)
    per_class_frame = pd.DataFrame(per_class_rows)
    distributions_frame = pd.DataFrame(distribution_rows)
    rounds_frame.to_csv(run_dir / "round_metrics.csv", index=False)
    per_class_frame.to_csv(run_dir / "per_class_metrics.csv", index=False)
    distributions_frame.to_csv(run_dir / "prediction_distributions.csv", index=False)
    best = rounds_frame.loc[rounds_frame.validation_macro_f1.idxmax()]
    final = rounds_frame.iloc[-1]
    summary = {
        "variant": name,
        "seed": seed,
        "training_objective": "effective-number weighted cross-entropy" if class_weights is not None else "standard cross-entropy",
        "effective_beta": beta,
        "class_weight_normalisation": "arithmetic mean equals 1" if class_weights is not None else None,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "initial_global_state_sha256": initial_hash,
        "rounds": rounds,
        "learning_rate": 0.001,
        "local_epochs": 3,
        "batch_size": 32,
        "clients_per_round": 150,
        "features": "68 metadata features",
        "aggregation": "sample-count FedAvg",
        "held_out_evaluated": False,
        "model_selection_rule": "highest validation Macro F1; report round 25 separately",
        "best_round": int(best["round"]),
        "best_validation_accuracy": float(best["validation_accuracy"]),
        "best_validation_macro_f1": float(best["validation_macro_f1"]),
        "best_validation_weighted_f1": float(best["validation_weighted_f1"]),
        "best_predicted_class_coverage": int(best["predicted_class_coverage"]),
        "round_25_validation_accuracy": float(final["validation_accuracy"]),
        "round_25_validation_macro_f1": float(final["validation_macro_f1"]),
        "round_25_validation_weighted_f1": float(final["validation_weighted_f1"]),
        "round_25_predicted_class_coverage": int(final["predicted_class_coverage"]),
        "runtime_seconds": float(final["elapsed_seconds"]),
    }
    (run_dir / "config_and_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, rounds_frame, per_class_frame, distributions_frame


def write_manifest(output: Path, experiment: dict) -> None:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append({"path": path.relative_to(output).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256(path)})
    (output / "artifact_manifest.json").write_text(
        json.dumps({"experiment": experiment, "artifact_count": len(artifacts), "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        experiment = json.loads((output / "experiment_configuration.json").read_text(encoding="utf-8"))
        write_manifest(output, experiment)
        print(f"Refreshed manifest for {output}")
        return

    train, validation, _held_out_unused, manifest = load_experiment_data()
    builder = ProposedFeatureBuilder().fit(train)
    training_labels = builder.category_encoder.transform(train.category)
    class_counts = np.bincount(training_labels, minlength=13)
    balanced_weights = effective_number_weights(training_labels, 13, beta=args.beta)
    class_table = pd.DataFrame({
        "category_index": np.arange(13),
        "category": builder.category_encoder.classes_,
        "training_count": class_counts,
        "effective_number_weight_mean_one": balanced_weights.numpy(),
    })
    class_table.to_csv(output / "training_class_counts_and_weights.csv", index=False)

    started = time.perf_counter()
    standard, standard_rounds, standard_classes, standard_distributions = run_variant(
        name="standard_cross_entropy_seed_42", class_weights=None, seed=args.seed,
        rounds=args.rounds, output=output, train=train, validation=validation,
        builder=builder, beta=None,
    )
    balanced, balanced_rounds, balanced_classes, balanced_distributions = run_variant(
        name="effective_number_seed_42", class_weights=balanced_weights, seed=args.seed,
        rounds=args.rounds, output=output, train=train, validation=validation,
        builder=builder, beta=args.beta,
    )
    if standard["initial_global_state_sha256"] != balanced["initial_global_state_sha256"]:
        raise AssertionError("Variants did not start from identical global parameters")

    key_rounds = [4, 6, 10, 15, 20, 25]
    all_rounds = pd.concat([standard_rounds, balanced_rounds], ignore_index=True)
    all_classes = pd.concat([standard_classes, balanced_classes], ignore_index=True)
    all_distributions = pd.concat([standard_distributions, balanced_distributions], ignore_index=True)
    all_rounds[all_rounds["round"].isin(key_rounds)].to_csv(output / "ROUND_COMPARISON.csv", index=False)
    all_classes.to_csv(output / "PER_CLASS_METRICS.csv", index=False)
    all_distributions.to_csv(output / "PREDICTION_DISTRIBUTIONS.csv", index=False)
    pd.DataFrame([standard, balanced]).to_csv(output / "summary.csv", index=False)
    experiment = {
        "seed": args.seed,
        "rounds": args.rounds,
        "beta": args.beta,
        "difference_under_test": "standard versus effective-number weighted local cross-entropy",
        "weight_formula": "(1-beta)/(1-beta**n_c), then divide by arithmetic mean",
        "class_counts_source": "training partition only",
        "validation_loss": "standard unweighted cross-entropy for both variants",
        "training_objective_losses_comparable": False,
        "matched": ["initial model", "architecture", "metadata features", "client map and order",
                    "participation", "reset Adam", "learning rate", "local epochs", "batch size",
                    "communication rounds", "sample-count FedAvg", "random seed"],
        "held_out_evaluated": False,
        "split_manifest_version": manifest.get("version"),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "experiment_configuration.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    write_manifest(output, experiment)
    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()
