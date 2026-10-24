
"""Validation-only ACTM versus count-matched Random replay.

Uses existing frozen unweighted Proposed checkpoints.
No training, refitting, threshold tuning or held-out evaluation.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.run_actm_selection_baselines import (
    load_checkpoint,
    outcome_rows,
    strategy_row,
)
from models.federated.actm import (
    ACTM,
    ACTMConfig,
    CrossAccountConflictDetector,
    budget_matched_selection_baselines,
)
from train_proposed import probabilities
from utils.experiment_data import load_experiment_data


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/ablations",
    )
    parser.add_argument(
        "--output",
        default="outputs/actm_validation_replay",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train, validation, _, manifest = load_experiment_data(
        "dataset/clean_budgetwise.csv",
        "data/experiment_split.json",
    )

    if len(train) != 7494 or len(validation) != 1729:
        raise ValueError("Frozen Tier A split does not match.")

    record_ids = validation["source_index"].to_numpy()
    available_notes = (
        validation["notes"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .to_numpy()
    )

    conflicts = (
        CrossAccountConflictDetector()
        .fit(train)
        .transform(validation)
    )

    run_rows = []
    comparison_rows = []
    trigger_rows = []
    transaction_frames = []
    checkpoint_hashes = {}

    for seed in (42, 52, 62):
        checkpoint = (
            Path(args.checkpoint_root)
            / f"seed_{seed}"
            / "full"
        )

        required = (
            "best_global_model.pt",
            "feature_pipeline.pkl",
            "model_config.json",
        )

        for name in required:
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(
                    f"Missing original checkpoint artifact: "
                    f"{checkpoint / name}. "
                    "Do not retrain a replacement model."
                )

        checkpoint_hashes[str(seed)] = {
            name: sha256(checkpoint / name)
            for name in required
        }

        features, model = load_checkpoint(checkpoint)

        metadata, notes, anchors, actual = (
            features.transform_parts(validation)
        )

        before = probabilities(
            model,
            features.metadata_only(metadata),
        )

        if before.shape != (1729, 13):
            raise ValueError(
                f"Unexpected probability shape: {before.shape}"
            )

        decisions = ACTM(
            ACTMConfig(0.65, 0.15, 0.30)
        ).decide(before, conflicts)

        actm_mask = decisions["triggered"].to_numpy()

        random_mask = budget_matched_selection_baselines(
            before,
            conflicts,
            record_ids,
            prompt_budget=0.30,
            seed=seed,
        )["Random"]

        m = int(actm_mask.sum())

        if m != int(random_mask.sum()):
            raise ValueError(
                "ACTM and Random selection counts differ."
            )

        if m > int(np.ceil(0.30 * len(validation))):
            raise ValueError("Prompt budget exceeded.")

        trigger_rows.append({
            "seed": seed,
            "records": len(validation),
            "entropy": int(
                decisions["entropy_triggered"].sum()
            ),
            "margin": int(
                decisions["margin_triggered"].sum()
            ),
            "conflict": int(
                decisions["conflict_triggered"].sum()
            ),
            "all_three": int((
                decisions["entropy_triggered"]
                & decisions["margin_triggered"]
                & decisions["conflict_triggered"]
            ).sum()),
            "eligible": int(decisions["eligible"].sum()),
            "selected": m,
        })

        seed_results = {}

        for strategy, prompted in {
            "Random": random_mask,
            "Proposed ACTM": actm_mask,
        }.items():

            note_used = prompted & available_notes

            fused = features.semantic_anchor_fused(
                metadata,
                notes,
                anchors,
                note_used,
            )

            after = probabilities(model, fused)

            row = strategy_row(
                seed,
                strategy,
                prompted,
                note_used,
                actual,
                before,
                after,
            )

            denominator = int(row["notes_used"])

            row["usable_source_note_rate"] = row.pop(
                "note_acceptance_rate"
            )
            row["selected_initially_wrong_rate"] = row.pop(
                "prompt_precision"
            )

            row["net_correction_rate"] = (
                row["net_corrections"] / denominator
                if denominator > 0
                else np.nan
            )

            if denominator == 0:
                row["mean_true_class_probability_change"] = (
                    np.nan
                )

            run_rows.append(row)
            seed_results[strategy] = row

            transaction_frames.append(
                outcome_rows(
                    seed,
                    strategy,
                    validation,
                    actual,
                    prompted,
                    note_used,
                    before,
                    after,
                )
            )

        random = seed_results["Random"]
        actm = seed_results["Proposed ACTM"]

        comparison_rows.append({
            "seed": seed,
            "random_usable": random["notes_used"],
            "actm_usable": actm["notes_used"],
            "random_net_rate": random["net_correction_rate"],
            "actm_net_rate": actm["net_correction_rate"],
            "delta_net_rate": (
                actm["net_correction_rate"]
                - random["net_correction_rate"]
            ),
            "delta_true_class_probability": (
                actm["mean_true_class_probability_change"]
                - random["mean_true_class_probability_change"]
            ),
        })

    runs = pd.DataFrame(run_rows)
    comparisons = pd.DataFrame(comparison_rows)

    pd.DataFrame(trigger_rows).to_csv(
        output / "validation_triggers.csv",
        index=False,
    )

    runs.to_csv(
        output / "validation_selection_runs.csv",
        index=False,
    )

    comparisons.to_csv(
        output / "validation_actm_vs_random.csv",
        index=False,
    )

    pd.concat(
        transaction_frames,
        ignore_index=True,
    ).to_csv(
        output / "validation_transaction_outcomes.csv",
        index=False,
    )

    summary = {
        "evaluation": "validation-only",
        "seeds": [42, 52, 62],
        "records_per_seed": 1729,
        "checkpoint_root": str(args.checkpoint_root),
        "checkpoint_sha256": checkpoint_hashes,
        "mean_delta_net_rate": (
            comparisons["delta_net_rate"].mean()
        ),
        "mean_delta_true_class_probability": (
            comparisons["delta_true_class_probability"].mean()
        ),
        "held_out_results_changed": False,
        "rq2_verdict_changed": False,
    }

    (output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nValidation ACTM vs Random")
    print(comparisons.to_string(index=False))
    print("\nMean differences:")
    print("Net rate:", summary["mean_delta_net_rate"])
    print(
        "True-class probability:",
        summary["mean_delta_true_class_probability"],
    )
    print("\nSaved to:", output)


if __name__ == "__main__":
    main()
