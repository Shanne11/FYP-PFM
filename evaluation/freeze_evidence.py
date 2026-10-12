"""Create a traceable manifest for the reportable Tier A evidence set."""

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_ARTIFACTS = (
    "dataset/clean_budgetwise.csv",
    "data/experiment_split.json",
    "deployment/research_model_contract.json",
    "outputs/comparison/baseline_comparison.csv",
    "outputs/repeated/repeated_runs.csv",
    "outputs/repeated/repeated_summary.csv",
    "outputs/ablations/ablation_runs.csv",
    "outputs/ablations/ablation_summary.csv",
    "outputs/actm_evaluation/actm_runs.csv",
    "outputs/actm_evaluation/actm_summary.csv",
    "outputs/actm_selection_baselines/actm_selection_runs.csv",
    "outputs/actm_selection_baselines/actm_selection_summary.csv",
    "outputs/actm_selection_baselines/actm_selection_contract.json",
    "outputs/actm_threshold_analysis/actm_threshold_distributions.csv",
    "outputs/actm_threshold_analysis/actm_threshold_sensitivity_summary.csv",
    "outputs/class_analysis/class_runs.csv",
    "outputs/class_analysis/class_summary.csv",
    "outputs/class_balance/class_balance_runs.csv",
    "outputs/class_balance/class_balance_summary.csv",
    "outputs/statistics/paired_statistics.csv",
    "outputs/utility_diagnostics/utility_diagnostic_summary.csv",
    "outputs/class_balance/seed_42/proposed/best_global_model.pt",
    "outputs/class_balance/seed_42/proposed/feature_pipeline.pkl",
    "deployment/pocketiq_deployment_package/deployment_metrics.json",
    "deployment/pocketiq_deployment_package/parity_report.json",
    "deployment/pocketiq_deployment_package/package_manifest.json",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default="outputs/evidence_freeze")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty evidence artifact: {path}")
    if path.suffix.lower() == ".json":
        json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            if next(reader, None) is None:
                raise ValueError(f"CSV has no header: {path}")


def git_value(root, *args):
    process = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=True
    )
    return process.stdout.strip()


def build_manifest(root):
    artifacts = []
    for relative in REQUIRED_ARTIFACTS:
        path = root / relative
        validate_artifact(path)
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    status = git_value(root, "status", "--porcelain")
    return {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": git_value(root, "rev-parse", "HEAD"),
        "working_tree_clean": not bool(status),
        "freeze_status": "final" if not status else "provisional-uncommitted-changes",
        "scope": "Tier A research summaries, selected checkpoint, contracts, and separate deployment verification",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main(config):
    root = config.root.resolve()
    output = (root / config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root)
    destination = output / "evidence_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = (
        "Tier A evidence-freeze verification\n\n"
        f"Status: {manifest['freeze_status']}\n"
        f"Repository HEAD: {manifest['repository_head']}\n"
        f"Validated and hashed artifacts: {manifest['artifact_count']}\n"
        "The manifest is final only after the intended code and evidence files are committed.\n"
    )
    (output / "evidence_freeze_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main(arguments())
