"""Validation-only reset versus per-client persistent Adam experiment."""

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


DEFAULT_OUTPUT = ROOT / "experiments" / "federated_diagnosis" / "outputs" / "adam_persistence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(results: list[dict]) -> dict[str, torch.Tensor]:
    counts = np.asarray([row["sample_count"] for row in results], dtype=float)
    coefficients = counts / counts.sum()
    result = copy.deepcopy(results[0]["weights"])
    for key in result:
        result[key] = result[key] * coefficients[0]
        for index in range(1, len(results)):
            result[key] += results[index]["weights"][key] * coefficients[index]
    return result


class PersistentAdamClient:
    """One client's model plus its private Adam moment/state history."""

    def __init__(self, model: nn.Module, learning_rate: float):
        self.model = model
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_global_parameters(self, global_state: dict[str, torch.Tensor]) -> None:
        # Loading parameters does not replace Parameter objects, so this client's
        # optimiser remains attached and retains only this client's moment state.
        self.model.load_state_dict(global_state)

    def train(self, features: torch.Tensor, targets: torch.Tensor, epochs: int = 3) -> float:
        loader = DataLoader(TensorDataset(features, targets), batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        total = 0.0
        observations = 0
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                self.optimiser.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                self.optimiser.step()
                total += float(loss.item()) * len(batch_y)
                observations += len(batch_y)
        return total / observations

    def optimiser_step(self) -> int:
        steps = []
        for state in self.optimiser.state.values():
            step = state.get("step")
            if step is not None:
                steps.append(int(step.item() if torch.is_tensor(step) else step))
        return max(steps, default=0)


def run_variant(
    *, name: str, persistent: bool, seed: int, rounds: int, learning_rate: float,
    output: Path, train: pd.DataFrame, validation: pd.DataFrame,
    builder: ProposedFeatureBuilder,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir = output / name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_x, _, _, train_y = builder.transform_parts(train)
    validation_x, _, _, validation_y = builder.transform_parts(validation)
    if train_x.shape[1] != 68 or len(builder.category_encoder.classes_) != 13:
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

    global_model = MLP(68, 13)
    initial_state_hash = hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in global_model.state_dict().values())
    ).hexdigest()
    persistent_clients = None
    if persistent:
        persistent_clients = {
            client_id: PersistentAdamClient(copy.deepcopy(global_model), learning_rate)
            for client_id in client_ids
        }

    criterion = nn.CrossEntropyLoss()
    validation_targets = torch.as_tensor(validation_y, dtype=torch.long)
    round_rows = []
    state_rows = []
    started = time.perf_counter()
    best_macro = -1.0
    for round_number in range(1, rounds + 1):
        global_state = copy.deepcopy(global_model.state_dict())
        client_results = []
        weighted_loss = 0.0
        for client_id in client_ids:
            features, targets = cache[client_id]
            if persistent:
                client = persistent_clients[client_id]
                client.load_global_parameters(global_state)
            else:
                client = PersistentAdamClient(copy.deepcopy(global_model), learning_rate)
            local_loss = client.train(features, targets, epochs=3)
            sample_count = len(targets)
            weighted_loss += local_loss * sample_count
            client_results.append({
                "weights": copy.deepcopy(client.model.state_dict()),
                "sample_count": sample_count,
            })
            state_rows.append({
                "variant": name, "round": round_number, "client_id": client_id,
                "optimiser_step": client.optimiser_step(), "persistent": persistent,
            })
        global_model.load_state_dict(aggregate(client_results))
        global_model.eval()
        with torch.no_grad():
            logits = global_model(torch.as_tensor(validation_x.toarray(), dtype=torch.float32))
        predictions = logits.argmax(dim=1).numpy()
        macro = f1_score(validation_y, predictions, average="macro", zero_division=0)
        row = {
            "variant": name,
            "seed": seed,
            "round": round_number,
            "persistent_client_adam_state": persistent,
            "training_loss": weighted_loss / len(train),
            "validation_loss": float(criterion(logits, validation_targets).item()),
            "validation_accuracy": accuracy_score(validation_y, predictions),
            "validation_macro_f1": macro,
            "predicted_class_coverage": int(np.unique(predictions).size),
            "single_class_collapse": bool(np.unique(predictions).size == 1),
            "elapsed_seconds": time.perf_counter() - started,
        }
        round_rows.append(row)
        if macro > best_macro:
            best_macro = macro
            torch.save(global_model.state_dict(), run_dir / "best_validation_model.pt")
        print(
            f"{name} round {round_number:02d}/{rounds}: "
            f"acc={row['validation_accuracy']:.4f} macro_f1={macro:.4f} "
            f"coverage={row['predicted_class_coverage']}"
        )

    metrics = pd.DataFrame(round_rows)
    states = pd.DataFrame(state_rows)
    metrics.to_csv(run_dir / "round_metrics.csv", index=False)
    states.to_csv(run_dir / "client_optimiser_steps.csv", index=False)
    best = metrics.loc[metrics.validation_macro_f1.idxmax()]
    final = metrics.iloc[-1]
    summary = {
        "variant": name,
        "seed": seed,
        "persistent_client_adam_state": persistent,
        "state_scope": "one private optimiser per client" if persistent else "new optimiser per client per round",
        "learning_rate": learning_rate,
        "rounds": rounds,
        "local_epochs": 3,
        "batch_size": 32,
        "clients_per_round": 150,
        "loss": "standard cross-entropy",
        "features": "68 metadata features",
        "initial_global_state_sha256": initial_state_hash,
        "held_out_evaluated": False,
        "best_round": int(best["round"]),
        "best_validation_accuracy": float(best["validation_accuracy"]),
        "best_validation_macro_f1": float(best["validation_macro_f1"]),
        "best_predicted_class_coverage": int(best["predicted_class_coverage"]),
        "round_10_validation_accuracy": float(metrics.loc[metrics["round"] == 10, "validation_accuracy"].iloc[0]),
        "round_10_validation_macro_f1": float(metrics.loc[metrics["round"] == 10, "validation_macro_f1"].iloc[0]),
        "round_10_predicted_class_coverage": int(metrics.loc[metrics["round"] == 10, "predicted_class_coverage"].iloc[0]),
        "round_25_validation_accuracy": float(final["validation_accuracy"]),
        "round_25_validation_macro_f1": float(final["validation_macro_f1"]),
        "round_25_predicted_class_coverage": int(final["predicted_class_coverage"]),
        "runtime_seconds": float(final["elapsed_seconds"]),
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
                "sha256": sha256(path),
            })
    (output / "artifact_manifest.json").write_text(
        json.dumps({"experiment": experiment, "artifact_count": len(artifacts), "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        write_manifest(output, experiment)
        print(f"Refreshed manifest for {output}")
        return

    train, validation, _held_out_unused, manifest = load_experiment_data()
    builder = ProposedFeatureBuilder().fit(train)
    started = time.perf_counter()
    reset = run_variant(
        name="reset_adam_seed_42", persistent=False, seed=args.seed, rounds=args.rounds,
        learning_rate=args.learning_rate, output=output, train=train,
        validation=validation, builder=builder,
    )
    persistent = run_variant(
        name="persistent_client_adam_seed_42", persistent=True, seed=args.seed,
        rounds=args.rounds, learning_rate=args.learning_rate, output=output,
        train=train, validation=validation, builder=builder,
    )
    if reset["initial_global_state_sha256"] != persistent["initial_global_state_sha256"]:
        raise AssertionError("Matched variants did not start from identical model parameters")
    pd.DataFrame([reset, persistent]).to_csv(output / "comparison.csv", index=False)
    experiment = {
        "seed": args.seed,
        "rounds": args.rounds,
        "learning_rate": args.learning_rate,
        "difference_under_test": "per-client Adam optimiser state reset versus persistence",
        "matched": [
            "initial global model", "architecture", "training and validation records",
            "68 metadata features", "150-client map and order", "all-client participation",
            "batch size 32", "three local epochs", "standard cross-entropy",
            "sample-count FedAvg", "random seed",
        ],
        "held_out_evaluated": False,
        "split_manifest_version": manifest.get("version"),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    write_manifest(output, experiment)
    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()
