"""Fixed-configuration Stage 5 held-out evaluation for BudgetWise.

The six Stage 4 configurations are deterministically replayed. Every validation
trajectory must reproduce the frozen Stage 4 evidence before the held-out test
features or labels are transformed and scored.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.federated_diagnosis.run_class_balance_screen import (
    aggregate,
    local_train,
    validation_statistics,
)
from models.mlp import MLP
from utils.class_balance import effective_number_weights
from utils.experiment_data import load_experiment_data
from utils.proposed_features import ProposedFeatureBuilder


STAGE3 = ROOT / "experiments" / "federated_diagnosis" / "outputs" / "class_balance_screen"
STAGE4 = ROOT / "experiments" / "federated_diagnosis" / "outputs" / "class_balance_replication"
DEFAULT_OUTPUT = ROOT / "experiments" / "federated_diagnosis" / "outputs" / "stage5_heldout"
SEEDS = (42, 52, 62)
LOSSES = ("standard", "effective_number")
ROUNDS = 25
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
BETA = 0.9999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def ordered_values_sha256(values) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_manifest(directory: Path) -> int:
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["artifacts"]:
        path = directory / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes", path.stat().st_size)
            or sha256(path) != item["sha256"]
        ):
            failures.append(item["path"])
    if failures:
        raise ValueError(f"Manifest verification failed for {directory}: {failures[:5]}")
    if manifest["artifact_count"] != len(manifest["artifacts"]):
        raise ValueError(f"Manifest count mismatch for {directory}")
    return len(manifest["artifacts"])


def stage4_contract() -> dict:
    verified_stage3 = verify_manifest(STAGE3)
    verified_stage4 = verify_manifest(STAGE4)
    stage3 = json.loads((STAGE3 / "experiment_configuration.json").read_text(encoding="utf-8"))
    stage4 = json.loads((STAGE4 / "experiment_configuration.json").read_text(encoding="utf-8"))
    required_stage3 = {
        "seed": 42,
        "rounds": ROUNDS,
        "beta": BETA,
        "held_out_evaluated": False,
        "split_manifest_version": 2,
    }
    for key, expected in required_stage3.items():
        if stage3.get(key) != expected:
            raise ValueError(f"Stage 3 configuration mismatch: {key}={stage3.get(key)!r}")
    required_stage4 = {
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "beta": BETA,
        "features": 68,
        "clients": 150,
        "held_out_evaluated": False,
        "split_manifest_version": 2,
    }
    for key, expected in required_stage4.items():
        if stage4.get(key) != expected:
            raise ValueError(f"Stage 4 configuration mismatch: {key}={stage4.get(key)!r}")
    if stage4.get("reused_seed_42_verified_artifacts") != verified_stage3:
        raise ValueError("Stage 4 does not record the currently verified Stage 3 artifact count")
    return {
        "stage3_verified_artifacts": verified_stage3,
        "stage4_verified_artifacts": verified_stage4,
        "stage3_configuration": stage3,
        "stage4_configuration": stage4,
    }


def previous_initial_hash(seed: int, loss: str) -> str:
    root = STAGE3 if seed == 42 else STAGE4
    name = (
        f"standard_cross_entropy_seed_{seed}"
        if loss == "standard"
        else f"effective_number_seed_{seed}"
    )
    summary = json.loads((root / name / "config_and_summary.json").read_text(encoding="utf-8"))
    return summary["initial_global_state_sha256"]


def train_fixed_round_25(
    *,
    seed: int,
    loss_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    builder: ProposedFeatureBuilder,
    class_weights: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame, str, float]:
    """Train without accepting any held-out data or held-out labels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_x, _, _, train_y = builder.transform_parts(train)
    validation_x, _, _, validation_y = builder.transform_parts(validation)
    if train_x.shape[1] != 68 or len(builder.category_encoder.classes_) != 13:
        raise ValueError("Expected the frozen 68-feature, 13-category metadata contract")

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
    expected_initial = previous_initial_hash(seed, loss_name)
    if initial_hash != expected_initial:
        raise ValueError(
            f"Initial model hash differs from Stage 4 evidence for seed={seed}, loss={loss_name}"
        )

    validation_tensor = torch.as_tensor(validation_x.toarray(), dtype=torch.float32)
    active_weights = None if loss_name == "standard" else class_weights
    rows = []
    started = time.perf_counter()
    for round_number in range(1, ROUNDS + 1):
        client_results = []
        weighted_objective_loss = 0.0
        for client_id in client_ids:
            features, targets = cache[client_id]
            local_model = copy.deepcopy(model)
            weights, objective_loss = local_train(
                local_model, features, targets, LEARNING_RATE, active_weights
            )
            sample_count = len(targets)
            weighted_objective_loss += objective_loss * sample_count
            client_results.append({"weights": weights, "sample_count": sample_count})
        model.load_state_dict(aggregate(client_results))
        model.eval()
        with torch.no_grad():
            logits = model(validation_tensor)
        statistics = validation_statistics(logits, validation_y)
        statistics.pop("predictions")
        rows.append(
            {
                "loss": loss_name,
                "seed": seed,
                "round": round_number,
                "training_objective_loss": weighted_objective_loss / len(train),
                **statistics,
            }
        )
        print(
            f"{loss_name} seed {seed} round {round_number:02d}/{ROUNDS}: "
            f"val_acc={statistics['validation_accuracy']:.4f} "
            f"val_macro_f1={statistics['validation_macro_f1']:.4f} "
            f"coverage={statistics['predicted_class_coverage']}",
            flush=True,
        )
    return copy.deepcopy(model.state_dict()), pd.DataFrame(rows), initial_hash, time.perf_counter() - started


def verify_stage4_trajectory(current: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_csv(STAGE4 / "ROUND_TRAJECTORIES.csv")
    keys = ["loss", "seed", "round"]
    columns = [
        "training_objective_loss",
        "validation_standard_cross_entropy",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_weighted_f1",
        "predicted_class_coverage",
    ]
    merged = current.merge(frozen[keys + columns], on=keys, suffixes=("_replay", "_stage4"), validate="one_to_one")
    if len(merged) != len(SEEDS) * len(LOSSES) * ROUNDS:
        raise ValueError("Stage 4 trajectory join is incomplete")
    checks = []
    for column in columns:
        delta = (merged[f"{column}_replay"] - merged[f"{column}_stage4"]).abs()
        maximum = float(delta.max())
        checks.append({"metric": column, "maximum_absolute_difference": maximum, "passed": maximum <= 1e-12})
    result = pd.DataFrame(checks)
    if not result.passed.all():
        raise ValueError(f"Stage 4 deterministic replay failed:\n{result}")
    return result


def evaluate_model(
    model: MLP,
    features: torch.Tensor,
    labels: np.ndarray,
    categories: list[str],
) -> tuple[dict, pd.DataFrame, np.ndarray, pd.DataFrame]:
    model.eval()
    with torch.no_grad():
        logits = model(features)
    predictions = logits.argmax(dim=1).cpu().numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(len(categories)), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "category_index": np.arange(len(categories)),
            "category": categories,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    counts = np.bincount(predictions, minlength=len(categories))
    distribution = pd.DataFrame(
        {
            "category_index": np.arange(len(categories)),
            "category": categories,
            "predicted_count": counts,
            "predicted_share": counts / len(predictions),
        }
    )
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "standard_cross_entropy": float(nn.CrossEntropyLoss()(logits, torch.as_tensor(labels, dtype=torch.long)).item()),
        "predicted_category_coverage": int(np.unique(predictions).size),
        "zero_recall_category_count": int((recall == 0).sum()),
        "zero_recall_categories": ";".join(per_class.loc[per_class.recall == 0, "category"]),
    }
    return metrics, per_class, predictions, distribution


def aggregate_rows(seed_frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "weighted_f1", "predicted_category_coverage", "zero_recall_category_count"]
    rows = []
    for loss_name in LOSSES:
        subset = seed_frame[seed_frame.loss == loss_name]
        row = {"comparison": loss_name, "seeds": "42,52,62", "runs": len(subset)}
        for metric in metrics:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_sample_std"] = subset[metric].std(ddof=1)
        rows.append(row)
    effective = seed_frame[seed_frame.loss == "effective_number"].set_index("seed")
    standard = seed_frame[seed_frame.loss == "standard"].set_index("seed")
    paired = {"comparison": "effective_number_minus_standard_paired", "seeds": "42,52,62", "runs": 3}
    for metric in metrics:
        values = effective[metric] - standard[metric]
        paired[f"{metric}_mean"] = values.mean()
        paired[f"{metric}_sample_std"] = values.std(ddof=1)
    rows.append(paired)
    return pd.DataFrame(rows)


def report_text(seed_frame: pd.DataFrame, aggregate: pd.DataFrame) -> str:
    effective = seed_frame[seed_frame.loss == "effective_number"].sort_values("seed")
    standard = seed_frame[seed_frame.loss == "standard"].sort_values("seed")
    effective_agg = aggregate[aggregate.comparison == "effective_number"].iloc[0]
    standard_agg = aggregate[aggregate.comparison == "standard"].iloc[0]
    paired_agg = aggregate[aggregate.comparison == "effective_number_minus_standard_paired"].iloc[0]
    accuracy_every = bool((effective.accuracy.to_numpy() > standard.accuracy.to_numpy()).all())
    macro_every = bool((effective.macro_f1.to_numpy() > standard.macro_f1.to_numpy()).all())
    zero_min = int(effective.zero_recall_category_count.min())
    zero_max = int(effective.zero_recall_category_count.max())
    rows = []
    for seed in SEEDS:
        for label, frame in (("Standard CE", standard), ("Effective number", effective)):
            item = frame[frame.seed == seed].iloc[0]
            rows.append(
                f"| {seed} | {label} | {item.accuracy:.4f} | {item.macro_f1:.4f} | "
                f"{item.weighted_f1:.4f} | {int(item.predicted_category_coverage)} | "
                f"{int(item.zero_recall_category_count)} | {item.zero_recall_categories} |"
            )
    gap_accuracy = float(effective_agg.accuracy_mean - 0.2382880274)
    gap_macro = float(effective_agg.macro_f1_mean - 0.1287218013)
    return f"""# Stage 5: Fixed-Configuration Held-Out Evaluation

## Scope and evidence boundary

This exploratory extension evaluates the frozen Stage 4 standard-CE and effective-number configurations on the existing BudgetWise held-out test partition. The partition had already been inspected in earlier Tier A work, so this is not a pristine independent confirmatory test. No held-out result was used to tune beta, architecture, learning rate, local epochs, communication rounds or model-selection rules.

Before held-out scoring, all 150 validation-round records from the six deterministic training replays matched the Stage 4 trajectories with maximum absolute difference no greater than 1e-12. Held-out features and labels were transformed only after all six fixed models completed and passed that replay check.

## Fixed round-25 held-out results

| Seed | Loss | Accuracy | Macro F1 | Weighted F1 | Coverage | Zero-recall count | Zero-recall categories |
|---:|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Three-seed means and sample standard deviations:

| Loss | Accuracy | Macro F1 | Weighted F1 | Coverage | Zero-recall categories |
|---|---:|---:|---:|---:|---:|
| Standard CE | {standard_agg.accuracy_mean:.4f} ± {standard_agg.accuracy_sample_std:.4f} | {standard_agg.macro_f1_mean:.4f} ± {standard_agg.macro_f1_sample_std:.4f} | {standard_agg.weighted_f1_mean:.4f} ± {standard_agg.weighted_f1_sample_std:.4f} | {standard_agg.predicted_category_coverage_mean:.2f} ± {standard_agg.predicted_category_coverage_sample_std:.2f} | {standard_agg.zero_recall_category_count_mean:.2f} ± {standard_agg.zero_recall_category_count_sample_std:.2f} |
| Effective number | {effective_agg.accuracy_mean:.4f} ± {effective_agg.accuracy_sample_std:.4f} | {effective_agg.macro_f1_mean:.4f} ± {effective_agg.macro_f1_sample_std:.4f} | {effective_agg.weighted_f1_mean:.4f} ± {effective_agg.weighted_f1_sample_std:.4f} | {effective_agg.predicted_category_coverage_mean:.2f} ± {effective_agg.predicted_category_coverage_sample_std:.2f} | {effective_agg.zero_recall_category_count_mean:.2f} ± {effective_agg.zero_recall_category_count_sample_std:.2f} |

Mean paired effective-number minus standard differences were {paired_agg.accuracy_mean:+.4f} accuracy, {paired_agg.macro_f1_mean:+.4f} Macro F1, {paired_agg.weighted_f1_mean:+.4f} Weighted F1 and {paired_agg.predicted_category_coverage_mean:+.2f} categories of coverage.

## Answers to the Stage 5 questions

1. **Accuracy in every seed:** {"Yes" if accuracy_every else "No"}. Effective-number accuracy was {"higher" if accuracy_every else "not higher"} in every paired seed.
2. **Macro F1 in every seed:** {"Yes" if macro_every else "No"}. Effective-number Macro F1 was {"higher" if macro_every else "not higher"} in every paired seed.
3. **Consistency with validation:** {"The direction is consistent for both accuracy and Macro F1 across all seeds" if accuracy_every and macro_every else "The held-out direction is not fully consistent with the all-seed validation improvement"}. Magnitudes and category-specific results remain partition-dependent.
4. **Unrecognised categories:** Effective-number retained zero recall for {zero_min}–{zero_max} of 13 categories across seeds; see `PER_CLASS_METRICS.csv` for names and values.
5. **Centralised comparison:** The earlier centralised metadata-only MLP benchmark was validation-only (accuracy about 0.2383, Macro F1 about 0.1287). The effective-number held-out mean differs by {gap_accuracy:+.4f} accuracy and {gap_macro:+.4f} Macro F1, but this is not a matched same-partition comparison and cannot establish that the federated model has reached the centralised baseline.
6. **Remaining classification problem:** Class balancing should be interpreted as reducing collapse only if the per-seed coverage and Macro F1 support that conclusion. The absolute Macro F1 and zero-recall counts show that further classifier/data diagnosis remains necessary; class imbalance is not the only demonstrated limitation.
7. **Smart Note readiness:** These results can identify a stronger base-classifier candidate, but they do not by themselves make BudgetWise suitable for a new Smart Note claim. Earlier evidence found the available note proxies weak and repetitive. A genuine Smart Note study still requires more informative clarification data and a separate protocol.

## Limitations and stopping decision

- Three seeds describe variability but do not prove statistical significance.
- The held-out partition was previously inspected, making this an exploratory extension.
- The comparison isolates the loss function under the frozen federated configuration; it does not evaluate ACTM, Smart Note fusion, RGSF, utility weighting, local learning or PocketIQ.
- No further tuning or Smart Note experiment was started after observing these results.
"""


def write_manifest(output: Path, experiment: dict) -> int:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {"experiment": experiment, "artifact_count": len(artifacts), "artifacts": artifacts}
    (output / "artifact_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(artifacts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    contract = stage4_contract()
    train, validation, held_out, split_manifest = load_experiment_data()
    if (len(train), len(validation), len(held_out)) != (7494, 1729, 2308):
        raise ValueError("Historical split counts changed")
    all_indices = [set(frame.source_index.astype(int)) for frame in (train, validation, held_out)]
    if all_indices[0] & all_indices[1] or all_indices[0] & all_indices[2] or all_indices[1] & all_indices[2]:
        raise ValueError("Frozen partitions are not disjoint")

    builder = ProposedFeatureBuilder().fit(train)
    train_metadata, _, _, train_labels = builder.transform_parts(train)
    if train_metadata.shape[1] != 68 or len(builder.category_encoder.classes_) != 13:
        raise ValueError("Frozen feature/taxonomy contract changed")
    class_weights = effective_number_weights(train_labels, 13, beta=BETA)
    class_counts = np.bincount(train_labels, minlength=13)
    if not np.isclose(float(class_weights.mean()), 1.0, atol=1e-7):
        raise ValueError("Effective-number weights are not arithmetic-mean-1 normalised")
    clients = sorted(train.user_id.fillna("Unknown").astype(str).unique())
    if len(clients) != 150:
        raise ValueError("Frozen client map changed")

    trained_states = {}
    trajectory_frames = []
    checkpoint_rows = []
    runtime_rows = []
    started = time.perf_counter()
    for seed in SEEDS:
        for loss_name in LOSSES:
            state, trajectory, initial_hash, elapsed = train_fixed_round_25(
                seed=seed,
                loss_name=loss_name,
                train=train,
                validation=validation,
                builder=builder,
                class_weights=class_weights,
            )
            trained_states[(seed, loss_name)] = state
            trajectory_frames.append(trajectory)
            runtime_rows.append({"seed": seed, "loss": loss_name, "training_seconds": elapsed})
            checkpoint = output / "checkpoints" / f"{loss_name}_seed_{seed}_round_25.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint)
            checkpoint_rows.append(
                {
                    "seed": seed,
                    "loss": loss_name,
                    "round": 25,
                    "initial_global_state_sha256": initial_hash,
                    "round_25_state_sha256": state_sha256(state),
                    "checkpoint_file": checkpoint.relative_to(output).as_posix(),
                    "checkpoint_file_sha256": sha256(checkpoint),
                }
            )

    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    replay = verify_stage4_trajectory(trajectories)
    replay.to_csv(output / "STAGE4_REPRODUCTION_CHECK.csv", index=False)

    # Held-out data are transformed and labels accessed only after all six fixed
    # runs complete and their validation trajectories reproduce Stage 4.
    test_metadata, _, _, test_labels = builder.transform_parts(held_out)
    test_tensor = torch.as_tensor(test_metadata.toarray(), dtype=torch.float32)
    categories = list(builder.category_encoder.classes_)
    test_record_hash = ordered_values_sha256(held_out.source_index.astype(int).tolist())

    seed_rows = []
    per_class_frames = []
    distribution_frames = []
    confusion_rows = []
    predictions_by_seed = {}
    stage4_round25 = pd.read_csv(STAGE4 / "ROUND_TRAJECTORIES.csv")
    for seed in SEEDS:
        for loss_name in LOSSES:
            model = MLP(68, 13)
            model.load_state_dict(trained_states[(seed, loss_name)])
            metrics, per_class, predictions, distribution = evaluate_model(
                model, test_tensor, test_labels, categories
            )
            predictions_by_seed[(seed, loss_name)] = predictions
            seed_rows.append({"seed": seed, "loss": loss_name, "n_test": len(test_labels), "test_record_order_sha256": test_record_hash, **metrics})
            per_class.insert(0, "loss", loss_name)
            per_class.insert(0, "seed", seed)
            per_class_frames.append(per_class)
            distribution.insert(0, "loss", loss_name)
            distribution.insert(0, "seed", seed)
            distribution_frames.append(distribution)
            matrix = confusion_matrix(test_labels, predictions, labels=np.arange(13))
            for actual_index, actual in enumerate(categories):
                for predicted_index, predicted in enumerate(categories):
                    confusion_rows.append(
                        {
                            "seed": seed,
                            "loss": loss_name,
                            "actual_index": actual_index,
                            "actual_category": actual,
                            "predicted_index": predicted_index,
                            "predicted_category": predicted,
                            "count": int(matrix[actual_index, predicted_index]),
                        }
                    )

    seed_frame = pd.DataFrame(seed_rows).sort_values(["seed", "loss"]).reset_index(drop=True)
    effective = seed_frame[seed_frame.loss == "effective_number"].set_index("seed")
    standard = seed_frame[seed_frame.loss == "standard"].set_index("seed")
    for metric in ("accuracy", "macro_f1", "weighted_f1", "predicted_category_coverage", "zero_recall_category_count"):
        differences = effective[metric] - standard[metric]
        seed_frame[f"paired_effective_minus_standard_{metric}"] = seed_frame.seed.map(differences).where(
            seed_frame.loss == "effective_number"
        )
    seed_frame.to_csv(output / "SEED_COMPARISON.csv", index=False)

    aggregate = aggregate_rows(seed_frame)
    aggregate.to_csv(output / "AGGREGATE_METRICS.csv", index=False)

    per_class = pd.concat(per_class_frames, ignore_index=True)
    aggregate_class_rows = []
    for (loss_name, category_index, category), frame in per_class.groupby(["loss", "category_index", "category"], sort=False):
        for statistic, function in (("mean", "mean"), ("sample_std", "std")):
            aggregate_class_rows.append(
                {
                    "seed": statistic,
                    "loss": loss_name,
                    "category_index": category_index,
                    "category": category,
                    "precision": getattr(frame.precision, function)(ddof=1) if function == "std" else frame.precision.mean(),
                    "recall": getattr(frame.recall, function)(ddof=1) if function == "std" else frame.recall.mean(),
                    "f1": getattr(frame.f1, function)(ddof=1) if function == "std" else frame.f1.mean(),
                    "support": getattr(frame.support, function)(ddof=1) if function == "std" else frame.support.mean(),
                }
            )
    pd.concat([per_class, pd.DataFrame(aggregate_class_rows)], ignore_index=True).to_csv(
        output / "PER_CLASS_METRICS.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(output / "CONFUSION_MATRICES.csv", index=False)
    pd.concat(distribution_frames, ignore_index=True).to_csv(output / "PREDICTION_DISTRIBUTIONS.csv", index=False)

    gap_rows = []
    for _, item in seed_frame.iterrows():
        validation_row = stage4_round25[
            (stage4_round25.seed == int(item.seed))
            & (stage4_round25.loss == item.loss)
            & (stage4_round25["round"] == 25)
        ].iloc[0]
        gap_rows.append(
            {
                "seed": int(item.seed),
                "loss": item.loss,
                "validation_accuracy": validation_row.validation_accuracy,
                "test_accuracy": item.accuracy,
                "test_minus_validation_accuracy": item.accuracy - validation_row.validation_accuracy,
                "validation_macro_f1": validation_row.validation_macro_f1,
                "test_macro_f1": item.macro_f1,
                "test_minus_validation_macro_f1": item.macro_f1 - validation_row.validation_macro_f1,
                "validation_weighted_f1": validation_row.validation_weighted_f1,
                "test_weighted_f1": item.weighted_f1,
                "test_minus_validation_weighted_f1": item.weighted_f1 - validation_row.validation_weighted_f1,
                "validation_coverage": int(validation_row.predicted_class_coverage),
                "test_coverage": int(item.predicted_category_coverage),
                "test_minus_validation_coverage": int(item.predicted_category_coverage - validation_row.predicted_class_coverage),
            }
        )
    pd.DataFrame(gap_rows).to_csv(output / "VALIDATION_TEST_GAPS.csv", index=False)

    paired_rows = []
    for seed in SEEDS:
        standard_predictions = predictions_by_seed[(seed, "standard")]
        effective_predictions = predictions_by_seed[(seed, "effective_number")]
        for position, source_index in enumerate(held_out.source_index.astype(int)):
            paired_rows.append(
                {
                    "seed": seed,
                    "test_position": position,
                    "source_index": source_index,
                    "truth_index": int(test_labels[position]),
                    "truth_category": categories[int(test_labels[position])],
                    "standard_prediction_index": int(standard_predictions[position]),
                    "standard_prediction": categories[int(standard_predictions[position])],
                    "effective_prediction_index": int(effective_predictions[position]),
                    "effective_prediction": categories[int(effective_predictions[position])],
                }
            )
    pd.DataFrame(paired_rows).to_csv(output / "PAIRED_PREDICTIONS.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(output / "CHECKPOINT_HASHES.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output / "RUNTIME.csv", index=False)
    trajectories.to_csv(output / "REPLAYED_VALIDATION_TRAJECTORIES.csv", index=False)

    test_files = [
        "tests/test_stage5_heldout.py",
        "tests/test_class_balance_screen.py",
        "tests/test_class_balance_replication.py",
    ]
    test_command = [sys.executable, "-m", "pytest", *test_files, "-q"]
    test_run = subprocess.run(test_command, cwd=ROOT, text=True, capture_output=True)
    test_payload = {
        "command": test_command,
        "returncode": test_run.returncode,
        "stdout": test_run.stdout,
        "stderr": test_run.stderr,
        "passed": test_run.returncode == 0,
        "broader_suite_limitation": (
            "tests/test_frozen_evaluation_contracts.py was not included because its pre-existing "
            "import of models.federated.actm.matched_random_selection cannot currently be collected."
        ),
    }
    (output / "test_results.json").write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
    if test_run.returncode != 0:
        raise RuntimeError(f"Focused Stage 5 tests failed:\n{test_run.stdout}\n{test_run.stderr}")

    configuration = {
        "stage": 5,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_worktree_note": "diagnosis work and Stage 5 remain uncommitted by user instruction",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "dataset_sha256": sha256(ROOT / "dataset" / "clean_budgetwise.csv"),
        "split_manifest_sha256": sha256(ROOT / "data" / "experiment_split.json"),
        "split_manifest_version": split_manifest.get("version"),
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(held_out)},
        "test_record_order_sha256": test_record_hash,
        "client_map_sha256": ordered_values_sha256(train.user_id.fillna("Unknown").astype(str).tolist()),
        "category_order": categories,
        "training_class_counts": {categories[index]: int(count) for index, count in enumerate(class_counts)},
        "effective_number_weights": {categories[index]: float(value) for index, value in enumerate(class_weights)},
        "class_weight_normalisation": "arithmetic mean equals 1",
        "seeds": list(SEEDS),
        "features": 68,
        "architecture": "68 -> 128 -> 64 -> 13, ReLU, dropout 0.3",
        "optimiser": "client-local reset Adam",
        "learning_rate": LEARNING_RATE,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "clients_per_round": 150,
        "aggregation": "sample-count FedAvg",
        "communication_rounds": ROUNDS,
        "effective_number_beta": BETA,
        "primary_comparison": "fixed round 25",
        "selection_data": "validation only; no held-out tuning",
        "stage3_verified_artifacts": contract["stage3_verified_artifacts"],
        "stage4_verified_artifacts": contract["stage4_verified_artifacts"],
        "stage4_trajectory_reproduction_tolerance": 1e-12,
        "focused_tests_passed": True,
        "focused_test_count": 6,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "The test partition was inspected in prior Tier A work.",
            "Three seeds do not establish statistical significance.",
            "No hyperparameter was changed after held-out inspection.",
        ],
    }
    (output / "experiment_configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    (output / "STAGE5_HELDOUT_RESULTS.md").write_text(report_text(seed_frame, aggregate), encoding="utf-8")
    reproduction = r"""# Stage 5 reproduction

Run from the repository root using the existing pinned environment:

```powershell
.\.venv\Scripts\python.exe experiments\federated_diagnosis\run_stage5_heldout.py
.\.venv\Scripts\python.exe -m pytest tests\test_stage5_heldout.py tests\test_class_balance_screen.py tests\test_class_balance_replication.py -q
```

The output directory must be new or empty. The script verifies the Stage 3 and Stage 4 manifests, replays the fixed six training runs, requires exact Stage 4 validation-trajectory reproduction within `1e-12`, and only then evaluates the held-out test partition. Do not use Stage 5 results to tune the configuration.
"""
    (output / "REPRODUCTION.md").write_text(reproduction, encoding="utf-8")
    artifact_count = write_manifest(output, configuration)
    print(json.dumps({"output": str(output), "artifact_count": artifact_count, "configuration": configuration}, indent=2))


if __name__ == "__main__":
    main()
