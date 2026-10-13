"""Run the controlled effective-number and RGSF Tier A extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.smart_note_reliability.rgsf import (  # noqa: E402
    fused_probabilities, gate_mask, reliability_features, transition_counts, tune_gate,
)
from models.federated.actm import ACTM, ACTMConfig, CrossAccountConflictDetector  # noqa: E402
from models.mlp import MLP  # noqa: E402
from train_proposed import main as train_proposed, probabilities  # noqa: E402
from utils.experiment_data import load_experiment_data  # noqa: E402
from utils.metrics import metric_summary  # noqa: E402


SEEDS = (42, 52, 62)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/smart_note_reliability/results")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--effective-beta", type=float, default=0.9999)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--max-clients", type=int, default=None,
                        help="Smoke testing only; never use for reportable results.")
    return parser.parse_args()


def _config(output, seed, note_strategy, config, *, signal_adjustment=False):
    return SimpleNamespace(
        dataset="dataset/clean_budgetwise.csv",
        split_manifest="data/experiment_split.json",
        output=str(output), rounds=config.rounds, local_epochs=config.local_epochs,
        learning_rate=0.001, entropy_threshold=0.65, margin_threshold=0.15,
        prompt_budget=0.30, min_notes=1, max_clients=config.max_clients, seed=seed,
        note_strategy=note_strategy, fusion_mode="semantic_anchor",
        heuristic_weights=[0.5, 0.3, 0.2],
        disable_signal_adjustment=not signal_adjustment,
        class_weighted_loss=False, loss_mode="effective_number",
        effective_beta=config.effective_beta,
        conflict_merchant_column="location", conflict_account_column="payment_mode",
    )


def _load_model(path):
    builder = joblib.load(path / "feature_pipeline.pkl")
    model_config = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
    model = MLP(model_config["input_features"], model_config["output_classes"])
    model.load_state_dict(torch.load(path / "best_global_model.pt", map_location="cpu", weights_only=True))
    return builder, model


def _metrics(actual, probs):
    return metric_summary(actual, probs.argmax(1), probs, np.arange(probs.shape[1]))


def _evaluate_seed(seed, no_note_path, model_path, train, validation, test, output):
    no_note_builder, no_note_model = _load_model(no_note_path)
    builder, model = _load_model(model_path)
    detector = CrossAccountConflictDetector().fit(train)
    actm = ACTM(ACTMConfig(0.65, 0.15, 0.30))
    split_data = {}
    for split_name, frame in (("validation", validation), ("test", test)):
        metadata, notes, anchors, actual = builder.transform_parts(frame)
        base = probabilities(model, builder.metadata_only(metadata))
        decisions = actm.decide(base, detector.transform(frame))
        note_used = (
            decisions["triggered"].to_numpy()
            & frame["notes"].fillna("").str.strip().ne("").to_numpy()
        )
        enhanced = probabilities(
            model, builder.semantic_anchor_fused(metadata, notes, anchors, note_used)
        )
        features = reliability_features(base, enhanced, notes, anchors)
        split_data[split_name] = (
            frame, actual, base, enhanced, note_used, features, decisions
        )

    _, val_actual, val_base, val_note, val_used, val_features, _ = split_data["validation"]
    thresholds, search = tune_gate(
        val_actual, val_base, val_note, val_features, val_used
    )
    search.to_csv(output / "validation_gate_search.csv", index=False)

    frame, actual, base, enhanced, note_used, features, decisions = split_data["test"]
    no_note_metadata, _, _, no_note_actual = no_note_builder.transform_parts(test)
    if not np.array_equal(actual, no_note_actual):
        raise ValueError("A2 and A3 class order or held-out records do not match")
    no_note_probabilities = probabilities(
        no_note_model, no_note_builder.metadata_only(no_note_metadata)
    )
    accepted = gate_mask(features, note_used, thresholds)
    gated = fused_probabilities(base, enhanced, accepted)
    rows = []
    per_class_frames = []
    for variant, probs, accepted_mask in (
        ("A2_effective_number_without_notes", no_note_probabilities, np.zeros(len(base), dtype=bool)),
        ("A3_effective_number_unconditional_fusion", enhanced, note_used),
        ("A4_effective_number_RGSF", gated, accepted),
    ):
        predicted = probs.argmax(1)
        rows.append({
            "seed": seed, "variant": variant, **_metrics(actual, probs),
            "predicted_category_coverage": int(len(np.unique(probs.argmax(1)))),
            **transition_counts(
                actual,
                no_note_probabilities if variant.startswith("A2_") else base,
                probs,
                note_used,
                accepted_mask,
            ),
        })
        labels = builder.category_encoder.classes_
        per_class = pd.DataFrame(classification_report(
            actual, predicted, labels=np.arange(len(labels)), target_names=labels,
            output_dict=True, zero_division=0,
        )).T.reset_index(names="category")
        per_class.insert(0, "variant", variant)
        per_class.insert(0, "seed", seed)
        per_class_frames.append(per_class)
        pd.DataFrame(
            confusion_matrix(actual, predicted), index=labels, columns=labels
        ).to_csv(output / f"{variant}_confusion_matrix.csv")
    per_class_all = pd.concat(per_class_frames, ignore_index=True)
    per_class_all.to_csv(output / "per_class_all_variants.csv", index=False)
    transaction = pd.DataFrame({
        "seed": seed, "source_index": frame["source_index"],
        "transaction_id": frame["transaction_id"], "actual": actual,
        "base_prediction": base.argmax(1), "note_prediction": enhanced.argmax(1),
        "gated_prediction": gated.argmax(1), "actm_selected": decisions["triggered"],
        "note_available": frame["notes"].fillna("").str.strip().ne(""),
        "note_used": note_used, "gate_accepted": accepted,
        "gate_reason": np.where(
            ~note_used, "not_selected_or_note_unavailable",
            np.where(accepted, "accepted_all_thresholds", "retained_base_reliability_gate"),
        ),
        "base_confidence": base.max(1), "note_confidence": enhanced.max(1),
        "gated_confidence": gated.max(1),
    })
    transaction = pd.concat([transaction.reset_index(drop=True), features], axis=1)
    transaction.to_csv(output / "transaction_level_results.csv", index=False)
    (output / "gate_thresholds.json").write_text(
        json.dumps({"selected_on": "validation", **asdict(thresholds)}, indent=2),
        encoding="utf-8",
    )
    return rows, per_class_all


def _read_metrics(path):
    return pd.read_csv(path / "overall_metrics.csv").iloc[0].to_dict()


def _loss_comparison(root, seeds):
    rows = []
    contracts = (
        ("standard_cross_entropy", ROOT / "outputs" / "ablations", "full"),
        ("inverse_frequency", ROOT / "outputs" / "class_balance", "proposed"),
        ("effective_number", root, "A1_effective_number_matched_original"),
    )
    for seed in seeds:
        for loss, base, leaf in contracts:
            path = base / f"seed_{seed}" / leaf
            metrics = _read_metrics(path)
            predictions = pd.read_csv(path / "predictions.csv")
            actual_column = "actual_label" if "actual_label" in predictions else "actual"
            predicted_column = "predicted_label" if "predicted_label" in predictions else "predicted"
            actual_values = predictions[actual_column].astype(str)
            predicted_values = predictions[predicted_column].astype(str)
            label_values = sorted(actual_values.unique())
            zero_recall = sum(
                not ((actual_values == label) & (predicted_values == label)).any()
                for label in label_values
            )
            experiment = json.loads((path / "experiment_info.json").read_text(encoding="utf-8"))
            rows.append({
                "seed": seed, "loss": loss,
                "best_validation_macro_f1": experiment["best_validation_macro_f1"],
                "predicted_category_coverage": int(predicted_values.nunique()),
                "zero_recall_classes": int(zero_recall),
                **{key: metrics[key] for key in (
                    "accuracy", "macro_f1", "weighted_f1", "ece", "brier_score"
                )},
            })
    runs = pd.DataFrame(rows)
    summary = runs.groupby("loss", sort=False).agg(
        runs=("seed", "count"),
        validation_macro_f1_mean=("best_validation_macro_f1", "mean"),
        validation_macro_f1_std=("best_validation_macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"), weighted_f1_std=("weighted_f1", "std"),
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        brier_score_mean=("brier_score", "mean"), brier_score_std=("brier_score", "std"),
        predicted_category_coverage_mean=("predicted_category_coverage", "mean"),
        predicted_category_coverage_std=("predicted_category_coverage", "std"),
        zero_recall_classes_mean=("zero_recall_classes", "mean"),
        zero_recall_classes_std=("zero_recall_classes", "std"),
    ).reset_index()
    runs.to_csv(root / "loss_comparison_runs.csv", index=False)
    summary.to_csv(root / "loss_comparison_summary.csv", index=False)


def _existing_ablation_rows(seed, train, test):
    result = []
    full_path = ROOT / "outputs" / "ablations" / f"seed_{seed}" / "full"
    builder, model = _load_model(full_path)
    metadata, notes, anchors, actual = builder.transform_parts(test)
    before_probabilities = probabilities(model, builder.metadata_only(metadata))
    detector = CrossAccountConflictDetector().fit(train)
    decisions = ACTM(ACTMConfig(0.65, 0.15, 0.30)).decide(
        before_probabilities, detector.transform(test)
    )
    used = (
        decisions["triggered"].to_numpy()
        & test["notes"].fillna("").str.strip().ne("").to_numpy()
    )
    after_probabilities = probabilities(
        model, builder.semantic_anchor_fused(metadata, notes, anchors, used)
    )
    before = before_probabilities.argmax(1)
    after = after_probabilities.argmax(1)
    before_correct, after_correct = before == actual, after == actual
    changed = before != after
    common = {
        "note_used": int(used.sum()),
        "predicted_category_coverage": np.nan,
        "calibration_available": True,
    }
    for variant, leaf in (
        ("A0_existing_without_notes", "without_notes"),
        ("A1_existing_smart_note_fusion", "full"),
    ):
        metrics = _read_metrics(ROOT / "outputs" / "ablations" / f"seed_{seed}" / leaf)
        predictions = pd.read_csv(
            ROOT / "outputs" / "ablations" / f"seed_{seed}" / leaf / "predictions.csv"
        )
        is_notes = variant.startswith("A1_")
        result.append({
            "seed": seed, "variant": variant, **metrics, **common,
            "predicted_category_coverage": int(predictions["predicted_label"].nunique()),
            "accepted": int(used.sum()) if is_notes else 0,
            "rejected": 0 if is_notes else int(used.sum()),
            "acceptance_coverage": 1.0 if is_notes else 0.0,
            "wrong_to_correct": int((used & ~before_correct & after_correct).sum()) if is_notes else 0,
            "correct_to_wrong": int((used & before_correct & ~after_correct).sum()) if is_notes else 0,
            "wrong_to_different_wrong": int((used & ~before_correct & ~after_correct & changed).sum()) if is_notes else 0,
            "unchanged_correct": int((used & before_correct & after_correct & ~changed).sum()),
            "unchanged_incorrect": int((used & ~before_correct & ~after_correct & ~changed).sum()),
            "net_correct_change": int((used & after_correct).sum() - (used & before_correct).sum()) if is_notes else 0,
            "confident_but_wrong": int((
                used & (after_probabilities.max(1) > before_probabilities.max(1))
                & ~after_correct
            ).sum()) if is_notes else 0,
            "accepted_accuracy": float(after_correct[used].mean()) if is_notes and used.any() else np.nan,
        })
    return result


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_manifest(root):
    include = [
        root / "ablation_runs.csv", root / "ablation_summary.csv",
        root / "loss_comparison_runs.csv", root / "loss_comparison_summary.csv",
        root / "per_class_all_seeds.csv", root / "provenance.json",
    ]
    include.extend(sorted(root.glob("seed_*/RGSF_evaluation/gate_thresholds.json")))
    for seed_root in sorted(root.glob("seed_*")):
        evaluation = seed_root / "RGSF_evaluation"
        include.extend([
            evaluation / "A2_effective_number_without_notes_confusion_matrix.csv",
            evaluation / "A3_effective_number_unconditional_fusion_confusion_matrix.csv",
            evaluation / "A4_effective_number_RGSF_confusion_matrix.csv",
        ])
    records = []
    for path in include:
        if not path.exists():
            raise FileNotFoundError(f"Expected result artifact is missing: {path}")
        records.append({
            "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    (root / "artifact_manifest.json").write_text(
        json.dumps({"count": len(records), "artifacts": records}, indent=2),
        encoding="utf-8",
    )


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main(config):
    root = Path(config.output)
    root.mkdir(parents=True, exist_ok=True)
    train, validation, test, manifest = load_experiment_data()
    all_rows, class_rows = [], []
    for seed in config.seeds:
        seed_root = root / f"seed_{seed}"
        matched_path = seed_root / "A1_effective_number_matched_original"
        a2_path = seed_root / "A2_effective_number_without_notes"
        a3_path = seed_root / "A3_effective_number_unconditional_fusion"
        training_contracts = (
            (matched_path, "selective", True),
            (a2_path, "none", False),
            (a3_path, "selective", False),
        )
        for path, note_strategy, signal_adjustment in training_contracts:
            if not config.reuse or not (path / "best_global_model.pt").exists():
                train_proposed(_config(
                    path, seed, note_strategy, config,
                    signal_adjustment=signal_adjustment,
                ))
        result_path = seed_root / "RGSF_evaluation"
        result_path.mkdir(parents=True, exist_ok=True)
        rows, per_class = _evaluate_seed(
            seed, a2_path, a3_path, train, validation, test, result_path
        )
        all_rows.extend(_existing_ablation_rows(seed, train, test))
        all_rows.extend(rows); class_rows.append(per_class)

    runs = pd.DataFrame(all_rows)
    numeric = runs.select_dtypes(include="number").columns.difference(["seed"])
    summary_rows = []
    for variant, group in runs.groupby("variant", sort=False):
        row = {"variant": variant, "runs": len(group)}
        for column in numeric:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_std"] = group[column].std(ddof=1) if len(group) > 1 else 0.0
        summary_rows.append(row)
    runs.to_csv(root / "ablation_runs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(root / "ablation_summary.csv", index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(root / "per_class_all_seeds.csv", index=False)
    _loss_comparison(root, config.seeds)
    provenance = {
        "extension": "Class-Balanced Effective-Number Loss and Reliability-Gated Smart Note Fusion",
        "source_commit_before_extension": _git_head(),
        "dataset": "dataset/clean_budgetwise.csv",
        "dataset_sha256": _sha256("dataset/clean_budgetwise.csv"),
        "split_manifest": "data/experiment_split.json",
        "split_manifest_sha256": _sha256("data/experiment_split.json"),
        "split_counts": manifest["counts"], "split_seed": manifest["seed"],
        "training_seeds": list(config.seeds), "rounds": config.rounds,
        "local_epochs": config.local_epochs, "effective_beta": config.effective_beta,
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "selection_rule": "validation Macro F1; test labels untouched until final evaluation",
        "evidence_boundary": "Source notes are retrospective proxies, not real-user clarifications.",
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    _write_manifest(root)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main(arguments())
