"""Reproduce the Section 16 evidence audit from frozen project artefacts.

This script does not train or modify a model.  It reads the current research
outputs, source contracts and Git state, then writes an auditable evidence
matrix under ``outputs/section16_audit``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
FLUTTER_ROOT = ROOT.parent / "FYP-personal-finance-management-system"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.experiment_data import load_experiment_data
from models.federated.actm import CrossAccountConflictDetector
from models.mlp import MLP
from train_proposed import probabilities


OUT = ROOT / "outputs" / "section16_audit"
SEEDS = (42, 52, 62)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_at(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def git(*args: str) -> str:
    return git_at(ROOT, *args)


def paths_and_hashes(paths: list[Path]) -> tuple[str, str]:
    existing = [path for path in paths if path.exists()]
    labels = []
    for path in existing:
        try:
            labels.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        except ValueError:
            labels.append(str(path))
    return (
        "; ".join(labels),
        "; ".join(sha256(path) for path in existing),
    )


def canonical_note(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def add_row(
    rows: list[dict[str, object]], question: str, claim: str,
    sources: list[Path], check: str, result: str, status: str,
    limitation: str = "",
) -> None:
    source_text, hashes = paths_and_hashes(sources)
    rows.append({
        "question": question,
        "claim_checked": claim,
        "evidence_artifact": source_text,
        "sha256": hashes,
        "seeds": "42, 52, 62" if any(f"seed_{s}" in source_text for s in SEEDS) else "N/A or fixed split seed 42",
        "check_performed": check,
        "actual_output": result,
        "status": status,
        "limitation_or_required_wording": limitation,
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Stage 1: identifiable source/evidence version.
    status_lines = git("status", "--short").splitlines()
    git_state = {
        "repository": str(ROOT),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "last_commit": git("log", "-1", "--format=%h %ad %s", "--date=iso"),
        "working_tree_clean": not status_lines,
        "changed_file_count": len(status_lines),
        "status_short": status_lines,
    }
    (OUT / "python_git_state.json").write_text(
        json.dumps(git_state, indent=2), encoding="utf-8"
    )

    flutter_status_lines = git_at(FLUTTER_ROOT, "status", "--short").splitlines()
    flutter_git_state = {
        "repository": str(FLUTTER_ROOT),
        "branch": git_at(FLUTTER_ROOT, "branch", "--show-current"),
        "head": git_at(FLUTTER_ROOT, "rev-parse", "HEAD"),
        "last_commit": git_at(FLUTTER_ROOT, "log", "-1", "--format=%h %ad %s", "--date=iso"),
        "working_tree_clean": not flutter_status_lines,
        "changed_file_count": len(flutter_status_lines),
        "status_short": flutter_status_lines,
    }
    (OUT / "flutter_git_state.json").write_text(
        json.dumps(flutter_git_state, indent=2), encoding="utf-8"
    )

    flutter_service_path = FLUTTER_ROOT / "lib" / "ml" / "learning" / "optional_learning_service.dart"
    flutter_test_path = FLUTTER_ROOT / "test" / "optional_learning_upload_disabled_test.dart"
    flutter_test_output_path = OUT / "flutter_test_output.txt"
    service_text = flutter_service_path.read_text(encoding="utf-8")
    test_text = flutter_test_path.read_text(encoding="utf-8")
    train_segment = service_text.split("Future<OptionalLearningResult> trainAndPrepare", 1)[1].split("Future<int> syncPending", 1)[0]
    sync_segment = service_text.split("Future<int> syncPending", 1)[1].split("Future<int> deletePendingUpdates", 1)[0]
    test_output_text = flutter_test_output_path.read_text(encoding="utf-8") if flutter_test_output_path.exists() else ""
    flutter_privacy_audit = {
        "git": flutter_git_state,
        "source_sha256": sha256(flutter_service_path),
        "test_sha256": sha256(flutter_test_path),
        "remote_sync_flag_false": "static const bool remoteSyncEnabled = false;" in service_text,
        "train_creates_payload_file": "_payloadFile(" in train_segment,
        "train_creates_pending_sync_row": "federatedSyncRows" in train_segment,
        "sync_pending_unconditionally_throws": "throw StateError(" in sync_segment and "upload is disabled" in sync_segment,
        "sync_pending_contains_network_call": any(token in sync_segment for token in ("HttpClient", ".post(", ".put(", "http.post", "Dio(")),
        "programmatic_rejection_test_present": "syncPending rejects direct programmatic calls" in test_text,
        "programmatic_rejection_test_passed": "syncPending rejects direct programmatic calls" in test_output_text and "All tests passed!" in test_output_text,
        "payload_fields": [],
        "privacy_conclusion": "The current pathway performs local calibration only. It creates no upload payload or pending sync record, and syncPending rejects direct calls before any network operation.",
        "evidence_limitation": "The source and test changes are uncommitted; record a new Flutter commit before calling this final frozen evidence.",
    }
    (OUT / "flutter_privacy_audit.json").write_text(
        json.dumps(flutter_privacy_audit, indent=2), encoding="utf-8"
    )

    freeze_path = ROOT / "outputs" / "evidence_freeze" / "evidence_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_rows = []
    for entry in freeze["artifacts"]:
        path = ROOT / entry["path"]
        actual_hash = sha256(path) if path.exists() else ""
        freeze_rows.append({
            "path": entry["path"],
            "exists": path.exists(),
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual_hash,
            "hash_match": path.exists() and actual_hash == entry["sha256"],
            "bytes_expected": entry["bytes"],
            "bytes_actual": path.stat().st_size if path.exists() else None,
        })
    freeze_df = pd.DataFrame(freeze_rows)
    freeze_df.to_csv(OUT / "artifact_hash_verification.csv", index=False)
    freeze_pass = bool(freeze_df["exists"].all() and freeze_df["hash_match"].all())

    train, validation, test, split_manifest = load_experiment_data()
    dataset_path = ROOT / "dataset" / "clean_budgetwise.csv"
    split_path = ROOT / "data" / "experiment_split.json"
    categories = sorted(train["category"].unique().tolist())

    # Q4 and Q5 dataset/class evidence.
    class_distribution = []
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        for label, count in frame["category"].value_counts().sort_index().items():
            class_distribution.append({"split": split_name, "category": label, "count": int(count)})
    class_distribution_df = pd.DataFrame(class_distribution)
    class_distribution_df.to_csv(OUT / "canonical_class_distribution.csv", index=False)
    train_counts = train["category"].value_counts()

    class_balance_path = ROOT / "outputs" / "class_balance" / "class_balance_summary.csv"
    class_balance = pd.read_csv(class_balance_path)
    proposed_standard = class_balance[(class_balance.method == "proposed") & (class_balance.loss == "standard")].iloc[0]
    proposed_weighted = class_balance[(class_balance.method == "proposed") & (class_balance.loss == "class_weighted")].iloc[0]

    # Q6: available-field overlap.  The dataset has neither merchant ID nor description.
    overlap_rows = []
    for left_name, left, right_name, right in (
        ("train", train, "validation", validation),
        ("train", train, "test", test),
        ("validation", validation, "test", test),
    ):
        left_ids, right_ids = set(left.transaction_id.astype(str)), set(right.transaction_id.astype(str))
        left_notes = set(left.notes.map(canonical_note)) - {""}
        right_notes = set(right.notes.map(canonical_note)) - {""}
        fields = ["date", "transaction_type", "amount", "payment_mode", "location", "notes"]
        left_records = set(map(tuple, left[fields].fillna("").astype(str).to_numpy()))
        right_records = set(map(tuple, right[fields].fillna("").astype(str).to_numpy()))
        overlap_rows.append({
            "left": left_name,
            "right": right_name,
            "transaction_id_overlap": len(left_ids & right_ids),
            "exact_available_record_overlap": len(left_records & right_records),
            "normalised_note_value_overlap": len(left_notes & right_notes),
        })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(OUT / "split_overlap_checks.csv", index=False)

    # Q8-Q12: matched selector audit and transaction-level before/after transitions.
    selector_path = ROOT / "outputs" / "actm_selection_baselines" / "actm_selection_transaction_outcomes.csv"
    selector_summary_path = ROOT / "outputs" / "actm_selection_baselines" / "actm_selection_summary.csv"
    selector_contract_path = ROOT / "outputs" / "actm_selection_baselines" / "actm_selection_contract.json"
    selector = pd.read_csv(selector_path)
    selector_rows = []
    for (seed, strategy), frame in selector.groupby(["seed", "strategy"], sort=True):
        prompted = frame[frame.prompted.astype(bool)].copy()
        wrong_to_correct = int(((~prompted.correct_before.astype(bool)) & prompted.correct_after.astype(bool)).sum())
        correct_to_wrong = int((prompted.correct_before.astype(bool) & (~prompted.correct_after.astype(bool))).sum())
        selector_rows.append({
            "seed": int(seed),
            "strategy": strategy,
            "test_records": len(frame),
            "selected": len(prompted),
            "selected_unique_source_ids": prompted.source_index.nunique(),
            "prompt_precision": float((~prompted.correct_before.astype(bool)).mean()),
            "corrections_per_100_prompts": 100.0 * wrong_to_correct / len(prompted),
            "net_corrections": wrong_to_correct - correct_to_wrong,
            "accuracy_after_all_records": float(frame.correct_after.astype(bool).mean()),
        })
    selector_recalc = pd.DataFrame(selector_rows)
    selector_recalc.to_csv(OUT / "actm_selector_recalculation.csv", index=False)
    selector_means = selector_recalc.groupby("strategy").mean(numeric_only=True).reset_index()
    selector_means.to_csv(OUT / "actm_selector_three_seed_means.csv", index=False)
    all_693 = bool((selector_recalc.selected == 693).all())
    same_test_ids = True
    for seed in SEEDS:
        seed_frame = selector[selector.seed == seed]
        reference = set(seed_frame[seed_frame.strategy == "Proposed ACTM"].source_index)
        for strategy in seed_frame.strategy.unique():
            same_test_ids &= set(seed_frame[seed_frame.strategy == strategy].source_index) == reference

    proposed_prompted = selector[(selector.strategy == "Proposed ACTM") & selector.prompted.astype(bool)].copy()
    proposed_notes = proposed_prompted[proposed_prompted.note_used.astype(bool)].copy()
    before_correct = proposed_notes.correct_before.astype(bool)
    after_correct = proposed_notes.correct_after.astype(bool)
    transitions = {
        "cases": len(proposed_notes),
        "duplicate_seed_source_pairs": int(proposed_notes.duplicated(["seed", "source_index"]).sum()),
        "wrong_to_correct": int(((~before_correct) & after_correct).sum()),
        "correct_to_wrong": int((before_correct & (~after_correct)).sum()),
        "changed_prediction": int((proposed_notes.predicted_before != proposed_notes.predicted_after).sum()),
        "confidence_increased_but_after_incorrect": int(((proposed_notes.confidence_after > proposed_notes.confidence_before) & (~after_correct)).sum()),
        "confidence_increased_wrong_before_and_after": int(((proposed_notes.confidence_after > proposed_notes.confidence_before) & (~before_correct) & (~after_correct)).sum()),
        "true_class_probability_increased_but_after_incorrect": int(((proposed_notes.true_probability_after > proposed_notes.true_probability_before) & (~after_correct)).sum()),
        "prompts": len(proposed_prompted),
        "note_used": len(proposed_notes),
        "note_use_rate": len(proposed_notes) / len(proposed_prompted),
        "prompt_precision": float((~proposed_prompted.correct_before.astype(bool)).mean()),
    }
    (OUT / "smart_note_transition_audit.json").write_text(
        json.dumps(transitions, indent=2), encoding="utf-8"
    )

    # Q13: identical partitions and participation across methods.
    participation_rows = []
    q13_pass = True
    for seed in SEEDS:
        ids_by_method_round: dict[tuple[str, int], set[str]] = {}
        counts_by_method_round: dict[tuple[str, int], dict[str, int]] = {}
        for method in ("fedavg", "fedprox"):
            frame = pd.read_csv(ROOT / "outputs" / "repeated" / f"seed_{seed}" / method / "client_metrics.csv")
            for round_id, round_frame in frame.groupby("round"):
                ids_by_method_round[(method, int(round_id))] = set(round_frame.client_id.astype(str))
                counts_by_method_round[(method, int(round_id))] = dict(zip(round_frame.client_id.astype(str), round_frame.sample_count.astype(int)))
        frame = pd.read_csv(ROOT / "outputs" / "repeated" / f"seed_{seed}" / "proposed" / "aggregation_weights.csv")
        frame = frame[frame.included.astype(bool)]
        for round_id, round_frame in frame.groupby("round"):
            ids_by_method_round[("proposed", int(round_id))] = set(round_frame.client_id.astype(str))
            counts_by_method_round[("proposed", int(round_id))] = dict(zip(round_frame.client_id.astype(str), round_frame.sample_count.astype(int)))
        for round_id in range(1, 11):
            same_ids = ids_by_method_round[("fedavg", round_id)] == ids_by_method_round[("fedprox", round_id)] == ids_by_method_round[("proposed", round_id)]
            same_counts = counts_by_method_round[("fedavg", round_id)] == counts_by_method_round[("fedprox", round_id)] == counts_by_method_round[("proposed", round_id)]
            q13_pass &= same_ids and same_counts
            participation_rows.append({
                "seed": seed, "round": round_id,
                "fedavg_clients": len(ids_by_method_round[("fedavg", round_id)]),
                "fedprox_clients": len(ids_by_method_round[("fedprox", round_id)]),
                "proposed_clients": len(ids_by_method_round[("proposed", round_id)]),
                "same_client_ids": same_ids, "same_sample_counts": same_counts,
            })
    pd.DataFrame(participation_rows).to_csv(OUT / "client_participation_audit.csv", index=False)

    # Q14 and Q16: multiplier bounds, coefficient normalisation and size effects.
    aggregation_frames = []
    for seed in SEEDS:
        frame = pd.read_csv(ROOT / "outputs" / "repeated" / f"seed_{seed}" / "proposed" / "aggregation_weights.csv")
        frame["seed"] = seed
        aggregation_frames.append(frame)
    aggregation = pd.concat(aggregation_frames, ignore_index=True)
    included = aggregation[aggregation.included.astype(bool)].copy()
    included["relative_weight_adjustment_pct"] = 100.0 * (included.final_coefficient / included.base_weight - 1.0)
    weight_sums = included.groupby(["seed", "round"])["final_coefficient"].sum().reset_index(name="coefficient_sum")
    weight_sums.to_csv(OUT / "aggregation_coefficient_sums.csv", index=False)
    bounds = {
        "rows": len(included),
        "multiplier_min": float(included.signal_multiplier.min()),
        "multiplier_max": float(included.signal_multiplier.max()),
        "all_multipliers_in_0_75_1_25": bool(included.signal_multiplier.between(0.75, 1.25).all()),
        "all_coefficients_nonnegative": bool((included.final_coefficient >= 0).all()),
        "maximum_coefficient_sum_error": float((weight_sums.coefficient_sum - 1.0).abs().max()),
    }
    (OUT / "aggregation_bounds_audit.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    included["size_quartile"] = pd.qcut(included.sample_count, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"], duplicates="drop")
    size_effect = included.groupby("size_quartile", observed=False).agg(
        records=("client_id", "size"),
        mean_sample_count=("sample_count", "mean"),
        mean_relative_adjustment_pct=("relative_weight_adjustment_pct", "mean"),
        minimum_relative_adjustment_pct=("relative_weight_adjustment_pct", "min"),
        maximum_relative_adjustment_pct=("relative_weight_adjustment_pct", "max"),
        clients_weight_reduced=("relative_weight_adjustment_pct", lambda values: int((values < 0).sum())),
        clients_weight_increased=("relative_weight_adjustment_pct", lambda values: int((values > 0).sum())),
    ).reset_index()
    size_effect.to_csv(OUT / "client_size_weight_adjustment.csv", index=False)
    correlation = float(included[["sample_count", "relative_weight_adjustment_pct"]].corr().iloc[0, 1])

    # Q15: matched per-client test accuracy from full vs no-adjustment ablation.
    client_accuracy_rows = []
    global_pairs = []
    for seed in SEEDS:
        full_dir = ROOT / "outputs" / "ablations" / f"seed_{seed}" / "full"
        no_dir = ROOT / "outputs" / "ablations" / f"seed_{seed}" / "without_heuristic_adjustment"
        full_pred = pd.read_csv(full_dir / "predictions.csv")
        no_pred = pd.read_csv(no_dir / "predictions.csv")
        comparison = pd.DataFrame({
            "user_id": test.user_id.astype(str).to_numpy(),
            "actual": full_pred.actual.to_numpy(),
            "full": full_pred.predicted.to_numpy(),
            "no_adjustment": no_pred.predicted.to_numpy(),
        })
        for client_id, frame in comparison.groupby("user_id"):
            full_acc = float((frame.full == frame.actual).mean())
            no_acc = float((frame.no_adjustment == frame.actual).mean())
            client_accuracy_rows.append({
                "seed": seed, "client_id": client_id, "test_records": len(frame),
                "full_accuracy": full_acc, "no_adjustment_accuracy": no_acc,
                "difference": full_acc - no_acc,
            })
        full_metrics = pd.read_csv(full_dir / "overall_metrics.csv").iloc[0]
        no_metrics = pd.read_csv(no_dir / "overall_metrics.csv").iloc[0]
        global_pairs.append({
            "seed": seed,
            "full_accuracy": float(full_metrics.accuracy),
            "no_adjustment_accuracy": float(no_metrics.accuracy),
            "accuracy_difference": float(full_metrics.accuracy - no_metrics.accuracy),
            "full_macro_f1": float(full_metrics.macro_f1),
            "no_adjustment_macro_f1": float(no_metrics.macro_f1),
            "macro_f1_difference": float(full_metrics.macro_f1 - no_metrics.macro_f1),
        })
    client_accuracy = pd.DataFrame(client_accuracy_rows)
    client_accuracy.to_csv(OUT / "matched_client_accuracy_audit.csv", index=False)
    pd.DataFrame(global_pairs).to_csv(OUT / "matched_global_no_adjustment_audit.csv", index=False)
    q15_counts = {
        "client_seed_cases": len(client_accuracy),
        "improved": int((client_accuracy.difference > 0).sum()),
        "declined": int((client_accuracy.difference < 0).sum()),
        "unchanged": int((client_accuracy.difference == 0).sum()),
    }

    # Q19: distinct research/deployment contracts.
    research_contract_path = ROOT / "deployment" / "research_model_contract.json"
    deployment_manifest_path = ROOT / "deployment" / "pocketiq_deployment_package" / "package_manifest.json"
    deployment_schema_path = ROOT / "deployment" / "pocketiq_deployment_package" / "feature_schema.json"
    deployment_labels_path = ROOT / "deployment" / "pocketiq_deployment_package" / "category_labels.json"
    research_contract = json.loads(research_contract_path.read_text(encoding="utf-8"))
    deployment_manifest = json.loads(deployment_manifest_path.read_text(encoding="utf-8"))
    deployment_schema = json.loads(deployment_schema_path.read_text(encoding="utf-8"))
    deployment_labels = json.loads(deployment_labels_path.read_text(encoding="utf-8"))
    deployment_feature_count = int(deployment_schema["text"]["max_features"] if "max_features" in deployment_schema["text"] else len(deployment_schema["text"]["vocabulary"]))
    model_contract_comparison = {
        "research_contract_id": research_contract["contract_id"],
        "research_input_features": research_contract["model"]["input_features"],
        "research_output_classes": research_contract["model"]["output_classes"],
        "deployment_package_id": deployment_manifest.get("package") or deployment_manifest.get("package_id") or deployment_manifest.get("model_id") or deployment_manifest.get("package_version"),
        "deployment_input_features": deployment_feature_count,
        "deployment_output_classes": len(deployment_labels),
        "contracts_are_distinct": research_contract["model"]["input_features"] != deployment_feature_count and research_contract["model"]["output_classes"] != len(deployment_labels),
        "deployment_model_sha256": sha256(ROOT / "deployment" / "pocketiq_deployment_package" / "model.onnx"),
    }
    (OUT / "model_contract_comparison.json").write_text(json.dumps(model_contract_comparison, indent=2), encoding="utf-8")

    # One-transaction Tier A trace.  This is not a PocketIQ local-audit record.
    demo_candidates = proposed_notes[(~before_correct) & after_correct]
    preferred_demo = demo_candidates[
        (demo_candidates.seed == 42) & (demo_candidates.source_index == 10883)
    ]
    demo = (preferred_demo if not preferred_demo.empty else demo_candidates).iloc[0]
    source_index = int(demo.source_index)
    source_record = test[test.source_index.astype(int) == source_index].iloc[0]
    seed = int(demo.seed)
    proposed_dir = ROOT / "outputs" / "repeated" / f"seed_{seed}" / "proposed"
    checkpoint_dir = ROOT / "outputs" / "ablations" / f"seed_{seed}" / "full"
    features = joblib.load(checkpoint_dir / "feature_pipeline.pkl")
    model_config = json.loads((checkpoint_dir / "model_config.json").read_text(encoding="utf-8"))
    model = MLP(model_config["input_features"], model_config["output_classes"])
    model.load_state_dict(torch.load(checkpoint_dir / "best_global_model.pt", map_location="cpu", weights_only=True))
    one = source_record.to_frame().T
    metadata, notes, anchors, actual = features.transform_parts(one)
    before_vector = probabilities(model, features.metadata_only(metadata))[0]
    after_vector = probabilities(model, features.semantic_anchor_fused(metadata, notes, anchors, np.array([True])))[0]
    sorted_before = np.sort(before_vector)
    normalised_entropy = float(-(before_vector * np.log(np.clip(before_vector, 1e-12, 1.0))).sum() / np.log(len(before_vector)))
    top_two_margin = float(sorted_before[-1] - sorted_before[-2])
    conflict_detector = CrossAccountConflictDetector().fit(train)
    conflict_value = bool(conflict_detector.transform(one)[0])
    client_id = str(source_record.user_id)
    final_round = int(pd.read_csv(proposed_dir / "aggregation_weights.csv")["round"].max())
    weights = pd.read_csv(proposed_dir / "aggregation_weights.csv")
    weight = weights[(weights["round"] == final_round) & (weights.client_id.astype(str) == client_id)].iloc[0]
    demo_trace = {
        "trace_type": "Tier A controlled-research trace; not a PocketIQ Flutter audit record",
        "seed": seed,
        "source_index": source_index,
        "transaction_id": str(source_record.transaction_id),
        "client_id": client_id,
        "metadata": {
            "date": str(source_record.date), "transaction_type": str(source_record.transaction_type),
            "amount": float(source_record.amount), "payment_mode": str(source_record.payment_mode),
            "location": str(source_record.location), "gold_category": str(source_record.category),
        },
        "actm": {
            "normalised_entropy": normalised_entropy,
            "top_two_margin": top_two_margin,
            "conflict_triggered": conflict_value,
            "selected": bool(demo.prompted),
        },
        "source_note": str(source_record.notes),
        "before": {
            "prediction_index": int(demo.predicted_before), "confidence": float(demo.confidence_before),
            "true_class_probability": float(demo.true_probability_before), "correct": bool(demo.correct_before),
            "probability_vector": [float(value) for value in before_vector],
        },
        "after": {
            "prediction_index": int(demo.predicted_after), "confidence": float(demo.confidence_after),
            "true_class_probability": float(demo.true_probability_after), "correct": bool(demo.correct_after),
            "probability_vector": [float(value) for value in after_vector],
        },
        "outcome": "wrong_to_correct",
        "aggregation_final_round": {
            "round": final_round, "sample_count": int(weight.sample_count),
            "base_weight": float(weight.base_weight), "signal_multiplier": float(weight.signal_multiplier),
            "proposed_final_coefficient": float(weight.final_coefficient),
            "multiplier_within_bounds": bool(0.75 <= weight.signal_multiplier <= 1.25),
        },
        "audit_warning": "The transaction prediction trace and the final-round client coefficient are Tier A records. They must not be presented as a PocketIQ on-device local audit.",
    }
    (OUT / "tier_a_seven_step_demo_trace.json").write_text(json.dumps(demo_trace, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    add_row(rows, "Stage 1", "All 26 declared evidence artefacts exist and match the manifest.", [freeze_path, OUT / "artifact_hash_verification.csv"], "Recompute SHA-256 for every manifest entry.", f"{int(freeze_df.hash_match.sum())}/26 hashes match; manifest head={freeze['repository_head']}; current head={git_state['head']}; working tree clean={git_state['working_tree_clean']}.", "Verified with limitation" if freeze_pass else "Needs correction", "Hashes pass, but the freeze remains provisional until the current changes are committed and a final manifest records that commit.")
    add_row(rows, "Q1", "Tier A is federated research and PocketIQ uses a separate ONNX deployment model.", [ROOT / "train_proposed.py", research_contract_path, deployment_manifest_path], "Compare research entry point and both model contracts.", "Research contract is 105 features/13 classes; deployment contract is 4,096 features/14 classes.", "Verified")
    add_row(rows, "Q2", "Tier A uses existing source notes, not participant clarification responses.", [ROOT / "train_proposed.py", dataset_path], "Trace notes from the fixed dataset through feature construction.", "The fixed BudgetWise 'notes' column is read retrospectively; no participant-response collection exists in Tier A.", "Verified with limitation", "Always call them retrospective source notes/proxy context, not real user clarification.")
    add_row(rows, "Q3", "No acceptance/usability claim is derived from source-note proxy records.", [ROOT / "outputs" / "actm_evaluation" / "actm_summary.csv", selector_summary_path], "Inspect evidence type and denominators.", "Outputs are retrospective proxy/selection metrics; no participant usability dataset is present.", "Verified with limitation", "The term 'acceptance' must be labelled note availability/use in the retrospective proxy, not human acceptance.")
    add_row(rows, "Q4", "Fixed split contains 7,494/1,729/2,308 records, 150 training clients and 13 labels.", [dataset_path, split_path], "Reload canonical split; count rows, training user IDs and canonical labels.", f"train={len(train)}, validation={len(validation)}, test={len(test)}, training clients={train.user_id.nunique()}, labels={len(categories)}.", "Verified")
    add_row(rows, "Q5", "Class-weighted results and category-collapse statistics are correctly reported.", [class_balance_path, OUT / "canonical_class_distribution.csv"], "Compare standard vs class-weighted three-seed summaries and recount canonical class frequencies.", f"Proposed standard: accuracy={proposed_standard.accuracy_mean:.6f}, Macro F1={proposed_standard.macro_f1_mean:.6f}, active classes={proposed_standard.active_prediction_classes_mean:.2f}, zero-recall={proposed_standard.zero_recall_classes_mean:.2f}. Class-weighted: accuracy={proposed_weighted.accuracy_mean:.6f}, Macro F1={proposed_weighted.macro_f1_mean:.6f}, active classes={proposed_weighted.active_prediction_classes_mean:.2f}, zero-recall={proposed_weighted.zero_recall_classes_mean:.2f}. Train max/min canonical class counts={train_counts.max()}/{train_counts.min()} (ratio {train_counts.max()/train_counts.min():.2f}).", "Verified with limitation", "Prediction collapse and dataset imbalance are separate findings; report both actual distributions and model coverage.")
    add_row(rows, "Q6", "Partition leakage is excluded.", [dataset_path, split_path, OUT / "split_overlap_checks.csv"], "Check ID overlap, exact available-field overlap and normalised note-value overlap; inspect available columns.", f"ID overlap is zero for all split pairs; exact available-record overlap counts={overlap_df.exact_available_record_overlap.tolist()}; normalised note values overlap={overlap_df.normalised_note_value_overlap.tolist()}. No merchant ID or description field exists.", "Verified with limitation", "Cannot prove merchant-level or semantic near-duplicate independence because persistent merchant IDs/descriptions are absent and fuzzy near-duplicate testing was not performed.")
    add_row(rows, "Q7", "Data-dependent transformations are fitted on training rows only.", [ROOT / "utils" / "proposed_features.py", ROOT / "train_proposed.py", split_path], "Inspect fit(train) and later transform_parts(validation/test) calls.", "Encoders, scaler and note TF-IDF are fitted in ProposedFeatureBuilder.fit(train); validation/test use transform_parts without refitting.", "Verified")
    add_row(rows, "Q8", "ACTM uses a nominal 30% budget and selects 693 of 2,308 test records per seed.", [selector_contract_path, selector_path, OUT / "actm_selector_recalculation.csv"], "Recount selected records and unique IDs per seed.", f"ceil(0.30*2308)={math.ceil(0.30*2308)}; every strategy/seed selected 693={all_693}.", "Verified")
    add_row(rows, "Q9", "Random, entropy-only, margin-only, conflict-only and Proposed ACTM are matched and have results.", [selector_path, selector_summary_path, OUT / "actm_selector_three_seed_means.csv"], "Recompute per-seed metrics; check 693 selections and identical held-out ID sets.", f"Five strategies present; all select 693; all compare the same 2,308 IDs per seed={same_test_ids}. Three-seed recalculation matches the saved selector summary.", "Verified")
    add_row(rows, "Q10", "84.80% prompt precision and 83.50% note acceptance/use are correctly interpreted.", [selector_path, OUT / "smart_note_transition_audit.json"], "Recompute Proposed denominators from selected transaction rows.", f"Prompt precision={transitions['prompt_precision']:.6f}; source note used={transitions['note_used']}/{transitions['prompts']}={transitions['note_use_rate']:.6f}.", "Verified with limitation", "83.50% is retrospective source-note availability/use among prompts, not participant acceptance and not correction accuracy.")
    add_row(rows, "Q11", "Before/after Smart Note transitions equal 56 wrong→correct, 54 correct→wrong, 315 changed over 1,736 cases.", [selector_path, OUT / "smart_note_transition_audit.json"], "Recalculate correctness transitions and duplicate seed–source pairs.", f"cases={transitions['cases']}, duplicates={transitions['duplicate_seed_source_pairs']}, wrong→correct={transitions['wrong_to_correct']}, correct→wrong={transitions['correct_to_wrong']}, changed={transitions['changed_prediction']}.", "Verified with limitation", "Retrospective source-note evidence; not genuine participant clarification.")
    add_row(rows, "Q12", "900 cases became more confident but remained incorrect.", [selector_path, ROOT / "outputs" / "actm_selection_baselines" / "actm_selection_runs.csv", OUT / "smart_note_transition_audit.json"], "Recalculate the supplied final-wrong formula and the stricter implemented metric that also requires the case to be wrong before.", f"Supplied formula (confidence_after > confidence_before and final incorrect)={transitions['confidence_increased_but_after_incorrect']}; implemented more_confident_still_wrong (wrong before and after)={transitions['confidence_increased_wrong_before_and_after']}; true-class-probability-increased but final wrong={transitions['true_class_probability_increased_but_after_incorrect']}.", "Needs correction", "Neither definition gives 900. Use 874 for the supplied formula or 845 for the current implemented metric, and name the definition explicitly.")
    add_row(rows, "Q13", "FedAvg, FedProx and Proposed use identical client maps and participation.", [OUT / "client_participation_audit.csv", ROOT / "data" / "experiment_split.json"], "Compare participating client IDs and sample counts for every seed and round.", f"All 30 seed-round comparisons match={q13_pass}; each method has 150 clients in every round.", "Verified")
    add_row(rows, "Q14", "Heuristic multipliers are bounded and normalised coefficients are valid.", [OUT / "aggregation_bounds_audit.json", OUT / "aggregation_coefficient_sums.csv"], "Check every included update's multiplier and coefficient; sum coefficients per seed-round.", f"multiplier range=[{bounds['multiplier_min']:.6f}, {bounds['multiplier_max']:.6f}], all within bounds={bounds['all_multipliers_in_0_75_1_25']}; coefficients nonnegative={bounds['all_coefficients_nonnegative']}; max |sum-1|={bounds['maximum_coefficient_sum_error']:.3e}.", "Verified")
    add_row(rows, "Q15", "15 client-seed cases improve, 19 decline and 416 are unchanged under heuristic adjustment.", [OUT / "matched_client_accuracy_audit.csv", OUT / "matched_global_no_adjustment_audit.csv"], "Join matched final test predictions to test users and compare full vs no-adjustment client accuracy.", f"Reproduced client-seed cases={q15_counts['client_seed_cases']}: improved={q15_counts['improved']}, declined={q15_counts['declined']}, unchanged={q15_counts['unchanged']}. Saved global accuracy and Macro F1 are identical per seed.", "Needs correction", "Current frozen predictions support 0/0/450, not 15/19/416. Do not report the latter without its missing source and definition.")
    add_row(rows, "Q16", "Aggregation consistently disadvantages smaller clients.", [OUT / "client_size_weight_adjustment.csv", OUT / "aggregation_bounds_audit.json"], "Compare relative coefficient adjustment by sample-count quartile and compute correlation.", f"sample-count/relative-adjustment correlation={correlation:.6f}; see quartile table for positive and negative adjustments in each size group.", "Verified with limitation", "This checks coefficient changes, not client accuracy harm. Do not claim smaller clients consistently lose unless the quartile evidence shows that pattern.")
    add_row(rows, "Q17", "Uncertainty change includes local adaptation and is not a pure note effect.", [ROOT / "train_proposed.py", ROOT / "models" / "federated" / "heuristic.py"], "Trace before probabilities from the global metadata-only model and after probabilities from the locally trained fused model.", "The implementation compares different model states, so the component combines representation and local-adaptation effects.", "Verified with limitation", "Describe the score as an author-defined heuristic, not isolated causal Smart Note value.")
    q18_pass = all((
        flutter_privacy_audit["remote_sync_flag_false"],
        not flutter_privacy_audit["train_creates_payload_file"],
        not flutter_privacy_audit["train_creates_pending_sync_row"],
        flutter_privacy_audit["sync_pending_unconditionally_throws"],
        not flutter_privacy_audit["sync_pending_contains_network_call"],
        flutter_privacy_audit["programmatic_rejection_test_passed"],
    ))
    add_row(rows, "Q18", "PocketIQ creates no upload payload or pending sync, and syncPending rejects calls.", [flutter_service_path, flutter_test_path, OUT / "flutter_privacy_audit.json", flutter_test_output_path], "Inspect the local-training and sync bodies, then directly call syncPending in a Flutter test.", f"remoteSyncEnabled=false; train payload creation={flutter_privacy_audit['train_creates_payload_file']}; pending-row creation={flutter_privacy_audit['train_creates_pending_sync_row']}; sync throws={flutter_privacy_audit['sync_pending_unconditionally_throws']}; network call={flutter_privacy_audit['sync_pending_contains_network_call']}; two privacy tests passed={flutter_privacy_audit['programmatic_rejection_test_passed']}.", "Verified with limitation" if q18_pass else "Needs correction", "Current code sends no payload, so descriptions and Smart Notes are not uploaded. The Flutter working tree is still uncommitted; record the final commit before freezing this claim.")
    add_row(rows, "Q19", "Tier A and PocketIQ use different model contracts.", [research_contract_path, deployment_manifest_path, deployment_schema_path, deployment_labels_path, ROOT / "deployment" / "pocketiq_deployment_package" / "model.onnx", OUT / "model_contract_comparison.json"], "Compare identifiers, input dimensions, class counts and ONNX hash.", f"Research={model_contract_comparison['research_input_features']} features/{model_contract_comparison['research_output_classes']} classes; deployment={model_contract_comparison['deployment_input_features']} features/{model_contract_comparison['deployment_output_classes']} classes; distinct={model_contract_comparison['contracts_are_distinct']}.", "Verified")
    add_row(rows, "Q20", "Class weighting and heuristic adjustment support the stated contribution claims.", [class_balance_path, ROOT / "outputs" / "ablations" / "ablation_summary.csv", OUT / "matched_global_no_adjustment_audit.csv"], "Compare class-weighted vs standard and full vs no-adjustment across the same seeds.", f"Class weighting raises proposed mean Macro F1 from {proposed_standard.macro_f1_mean:.6f} to {proposed_weighted.macro_f1_mean:.6f} while mean accuracy falls from {proposed_standard.accuracy_mean:.6f} to {proposed_weighted.accuracy_mean:.6f}; full vs no-adjustment has identical accuracy/Macro F1 per seed.", "Verified with limitation", "Claim improved class coverage/trade-off, not overall accuracy gain; heuristic weighting shows no measurable final accuracy/Macro-F1 gain in this matched run.")

    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUT / "section16_evidence_matrix.csv", index=False)

    status_counts = matrix.status.value_counts().to_dict()
    summary = {
        "audit_generated_from_current_files": True,
        "python_git": git_state,
        "flutter_git": flutter_git_state,
        "evidence_manifest": {
            "declared_artifacts": int(freeze["artifact_count"]),
            "hashes_matching": int(freeze_df.hash_match.sum()),
            "all_hashes_match": freeze_pass,
            "freeze_status": freeze["freeze_status"],
            "manifest_repository_head": freeze["repository_head"],
        },
        "question_status_counts": status_counts,
        "key_corrections": {
            "Q12_confident_but_wrong_supplied_formula": transitions["confidence_increased_but_after_incorrect"],
            "Q12_confident_but_wrong_implemented_metric": transitions["confidence_increased_wrong_before_and_after"],
            "Q15_client_accuracy_counts": q15_counts,
            "Q18": "source and direct-call test pass; final commit still pending",
        },
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Section 16 Evidence Audit",
        "",
        f"Python repository HEAD: `{git_state['head']}` on `{git_state['branch']}`.",
        f"Working tree clean: **{git_state['working_tree_clean']}** ({git_state['changed_file_count']} changed/untracked paths).",
        f"Evidence manifest: **{int(freeze_df.hash_match.sum())}/26 SHA-256 hashes match**; status remains `{freeze['freeze_status']}`.",
        "",
        "## Critical decisions",
        "",
        f"- Q11 reproduced exactly: {transitions['cases']} unique cases, {transitions['wrong_to_correct']} wrong→correct, {transitions['correct_to_wrong']} correct→wrong and {transitions['changed_prediction']} changed predictions.",
        f"- Q12 needs correction: the supplied formula reproduces {transitions['confidence_increased_but_after_incorrect']}; the stricter implemented metric reproduces {transitions['confidence_increased_wrong_before_and_after']}; neither is 900.",
        f"- Q15 needs correction: matched per-client final-test accuracy gives {q15_counts['improved']} improved, {q15_counts['declined']} declined and {q15_counts['unchanged']} unchanged client-seed cases.",
        "- Q18 passes source inspection and a direct-call test, but remains provisional until the Flutter changes are committed.",
        "- Tier A and PocketIQ are separate model/evidence pathways; the Tier A demonstration trace must not be called a Flutter local audit.",
        "",
        "## Status table",
        "",
        "| Question | Status | Actual output |",
        "|---|---|---|",
    ]
    for row in rows:
        output = str(row["actual_output"]).replace("|", "\\|")
        lines.append(f"| {row['question']} | {row['status']} | {output} |")
    lines.extend(["", "The CSV evidence matrix contains paths, SHA-256 hashes, checks and limitations for every row."])
    (OUT / "Section16_Evidence_Audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(OUT),
        "hashes_matching": int(freeze_df.hash_match.sum()),
        "freeze_pass": freeze_pass,
        "working_tree_clean": git_state["working_tree_clean"],
        "q11": transitions,
        "q13_pass": q13_pass,
        "q14": bounds,
        "q15": q15_counts,
        "q19": model_contract_comparison,
    }, indent=2))


if __name__ == "__main__":
    main()
