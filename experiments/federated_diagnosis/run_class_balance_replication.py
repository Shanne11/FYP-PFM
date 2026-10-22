"""Replicate the validation-only effective-number experiment for seeds 52/62."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.federated_diagnosis.run_class_balance_screen import run_variant
from utils.class_balance import effective_number_weights
from utils.experiment_data import load_experiment_data
from utils.proposed_features import ProposedFeatureBuilder


STAGE3 = ROOT / "experiments" / "federated_diagnosis" / "outputs" / "class_balance_screen"
DEFAULT_OUTPUT = ROOT / "experiments" / "federated_diagnosis" / "outputs" / "class_balance_replication"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["artifacts"]:
        path = root / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise ValueError(f"Stage 3 artifact verification failed: {failures[:3]}")
    return len(manifest["artifacts"])


def normalise_rounds(frame: pd.DataFrame, loss: str, seed: int) -> pd.DataFrame:
    result = frame.copy()
    result["loss"] = loss
    result["seed"] = seed
    return result[[
        "loss", "seed", "round", "training_objective", "training_objective_loss",
        "validation_standard_cross_entropy", "validation_accuracy", "validation_macro_f1",
        "validation_weighted_f1", "predicted_class_coverage", "single_class_collapse",
        "elapsed_seconds",
    ]]


def seed_summary(rounds: pd.DataFrame, per_class: pd.DataFrame, loss: str, seed: int) -> dict:
    peak = rounds.loc[rounds.validation_macro_f1.idxmax()]
    final = rounds.loc[rounds["round"] == 25].iloc[0]
    final_classes = per_class[per_class["round"] == 25]
    return {
        "loss": loss,
        "seed": seed,
        "peak_round": int(peak["round"]),
        "peak_validation_accuracy": float(peak["validation_accuracy"]),
        "peak_validation_macro_f1": float(peak["validation_macro_f1"]),
        "peak_predicted_class_coverage": int(peak["predicted_class_coverage"]),
        "round_25_accuracy": float(final["validation_accuracy"]),
        "round_25_macro_f1": float(final["validation_macro_f1"]),
        "round_25_weighted_f1": float(final["validation_weighted_f1"]),
        "round_25_predicted_class_coverage": int(final["predicted_class_coverage"]),
        "round_25_zero_recall_categories": int((final_classes.recall == 0).sum()),
        "any_single_class_collapse": bool(rounds.single_class_collapse.astype(bool).any()),
        "late_round_minimum_coverage": int(rounds.loc[rounds["round"] >= 16, "predicted_class_coverage"].min()),
    }


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
    parser.add_argument("--seeds", default="52,62")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        experiment = json.loads((output / "experiment_configuration.json").read_text(encoding="utf-8"))
        write_manifest(output, experiment)
        print(f"Refreshed manifest for {output}")
        return

    verified_stage3_artifacts = verify_manifest(STAGE3)
    stage3_config = json.loads((STAGE3 / "experiment_configuration.json").read_text(encoding="utf-8"))
    if stage3_config["seed"] != 42 or stage3_config["rounds"] != 25 or stage3_config["beta"] != 0.9999:
        raise ValueError("Seed-42 Stage 3 configuration is not equivalent to the replication contract")

    dataset = ROOT / "dataset" / "clean_budgetwise.csv"
    split = ROOT / "data" / "experiment_split.json"
    train, validation, _held_out_unused, manifest = load_experiment_data()
    builder = ProposedFeatureBuilder().fit(train)
    training_labels = builder.category_encoder.transform(train.category)
    weights = effective_number_weights(training_labels, 13, beta=0.9999)

    all_rounds = []
    all_classes = []
    all_distributions = []
    summaries = []

    # Reuse verified seed-42 Stage 3 CSV evidence.
    for loss, directory in (
        ("standard", "standard_cross_entropy_seed_42"),
        ("effective_number", "effective_number_seed_42"),
    ):
        rounds = pd.read_csv(STAGE3 / directory / "round_metrics.csv")
        classes = pd.read_csv(STAGE3 / directory / "per_class_metrics.csv")
        distributions = pd.read_csv(STAGE3 / directory / "prediction_distributions.csv")
        all_rounds.append(normalise_rounds(rounds, loss, 42))
        classes["loss"] = loss
        distributions["loss"] = loss
        all_classes.append(classes)
        all_distributions.append(distributions)
        summaries.append(seed_summary(rounds, classes, loss, 42))

    started = time.perf_counter()
    for seed in [int(value) for value in args.seeds.split(",")]:
        for loss, class_weights, name in (
            ("standard", None, f"standard_cross_entropy_seed_{seed}"),
            ("effective_number", weights, f"effective_number_seed_{seed}"),
        ):
            summary, rounds, classes, distributions = run_variant(
                name=name, class_weights=class_weights, seed=seed, rounds=25,
                output=output, train=train, validation=validation, builder=builder,
                beta=0.9999 if class_weights is not None else None,
            )
            all_rounds.append(normalise_rounds(rounds, loss, seed))
            classes["loss"] = loss
            distributions["loss"] = loss
            all_classes.append(classes)
            all_distributions.append(distributions)
            summaries.append(seed_summary(rounds, classes, loss, seed))

    trajectories = pd.concat(all_rounds, ignore_index=True)
    classes = pd.concat(all_classes, ignore_index=True)
    distributions = pd.concat(all_distributions, ignore_index=True)
    seed_frame = pd.DataFrame(summaries).sort_values(["seed", "loss"])
    effective = seed_frame[seed_frame.loss == "effective_number"].set_index("seed")
    standard = seed_frame[seed_frame.loss == "standard"].set_index("seed")
    for metric in ("round_25_accuracy", "round_25_macro_f1", "round_25_weighted_f1",
                   "round_25_predicted_class_coverage"):
        differences = effective[metric] - standard[metric]
        seed_frame[metric + "_paired_difference"] = seed_frame.seed.map(differences).where(
            seed_frame.loss == "effective_number"
        )
    seed_frame.to_csv(output / "SEED_COMPARISON.csv", index=False)

    metric_columns = ["round_25_accuracy", "round_25_macro_f1", "round_25_weighted_f1",
                      "round_25_predicted_class_coverage", "round_25_zero_recall_categories"]
    aggregate_rows = []
    for loss, frame in seed_frame.groupby("loss"):
        row = {"loss": loss, "seeds": "42,52,62", "runs": len(frame)}
        for metric in metric_columns:
            row[metric + "_mean"] = frame[metric].mean()
            row[metric + "_std"] = frame[metric].std(ddof=1)
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(output / "AGGREGATE_METRICS.csv", index=False)

    round25_classes = classes[classes["round"] == 25].copy()
    per_class_rows = []
    for (loss, category), frame in round25_classes.groupby(["loss", "category"], sort=False):
        for _, item in frame.sort_values("seed").iterrows():
            per_class_rows.append({"loss": loss, "category": category, "statistic": f"seed_{int(item.seed)}",
                                   "recall": item.recall, "f1": item.f1, "support": int(item.support)})
        per_class_rows.append({"loss": loss, "category": category, "statistic": "mean",
                               "recall": frame.recall.mean(), "f1": frame.f1.mean(),
                               "support": frame.support.mean()})
        per_class_rows.append({"loss": loss, "category": category, "statistic": "std",
                               "recall": frame.recall.std(ddof=1), "f1": frame.f1.std(ddof=1),
                               "support": frame.support.std(ddof=1)})
    pd.DataFrame(per_class_rows).to_csv(output / "PER_CLASS_RECALL_COMPARISON.csv", index=False)
    trajectories.to_csv(output / "ROUND_TRAJECTORIES.csv", index=False)

    reproduction = """# Reproduction\n\nFrom the repository root:\n\n```powershell\n.\\.venv\\Scripts\\python.exe experiments\\federated_diagnosis\\run_class_balance_replication.py --seeds 52,62\n```\n\nSeed 42 is reused only after verifying all Stage 3 manifest hashes and its configuration. The script does not transform, predict, select, or score the held-out test partition.\n"""
    (output / "REPRODUCTION.md").write_text(reproduction, encoding="utf-8")

    experiment = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_worktree_note": "diagnosis scripts and outputs are uncommitted by user instruction",
        "dataset_sha256": sha256(dataset),
        "split_manifest_sha256": sha256(split),
        "split_manifest_version": manifest.get("version"),
        "seeds": [42, 52, 62],
        "newly_executed_seeds": [int(value) for value in args.seeds.split(",")],
        "reused_seed_42_verified_artifacts": verified_stage3_artifacts,
        "rounds": 25,
        "beta": 0.9999,
        "features": 68,
        "clients": 150,
        "held_out_evaluated": False,
        "runtime_seconds_new_seeds": time.perf_counter() - started,
        "statistical_limitation": "Three seeds describe variability but do not establish statistical significance.",
    }
    (output / "experiment_configuration.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    write_manifest(output, experiment)
    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()
