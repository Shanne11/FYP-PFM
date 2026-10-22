
"""V2 ACTM trigger-quality diagnostic; validation only."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prepare_features_v2 import V2FeatureBuilder
from models.mlp import MLP
from models.federated.actm import (
    ACTM,
    ACTMConfig,
    CrossAccountConflictDetector,
    budget_matched_selection_baselines,
)
from utils.experiment_data import load_experiment_data
from train_proposed import probabilities, fuse


def main():
    train, validation, _, manifest = load_experiment_data(
        str(V2 / "data/clean_budgetwise_v2.csv"),
        str(V2 / "data/experiment_split_v2.json"),
    )

    assert manifest["group_key"] == "transaction_id"
    assert len(validation) == 2102

    checkpoint = (
        V2 / "outputs/validation/full/"
        "proposed_no_signal_adjustment_seed_42"
    )

    builder = joblib.load(
        checkpoint / "feature_pipeline_v2.joblib"
    )

    model = MLP(108, 13)
    model.load_state_dict(
        torch.load(
            checkpoint / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    meta, notes, anchors, actual = (
        builder.transform_parts(validation)
    )

    # The frozen model's metadata-only predictions.
    before = probabilities(
        model, builder.metadata_only(meta)
    )
    before_pred = before.argmax(axis=1)
    errors = before_pred != actual

    no_note_f1 = f1_score(
        actual, before_pred,
        average="macro", zero_division=0
    )

    # Training-fitted context conflicts.
    detector = CrossAccountConflictDetector().fit(train)
    conflicts = detector.transform(validation)

    actm = ACTM(ACTMConfig(0.65, 0.15, 0.30))
    decisions = actm.decide(before, conflicts)

    actm_mask = decisions["triggered"].to_numpy()
    budget = int(actm_mask.sum())
    fraction = budget / len(validation)

    # source_index is unique even where transaction_id
    # is duplicated within the validation partition.
    record_ids = (
        validation["source_index"]
        .astype(str)
        .to_numpy()
    )

    assert len(set(record_ids)) == len(validation)

    available = (
        validation["notes"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .to_numpy()
    )

    tfidf_nonzero = (
        np.asarray(notes.getnnz(axis=1)).ravel() > 0
    )

    def evaluate(name, selected, seed=None):
        selected = np.asarray(selected, dtype=bool)

        # All policies use the same number of
        # prompt opportunities.
        if int(selected.sum()) != budget:
            raise ValueError(
                f"{name}: prompt budget mismatch"
            )

        used = selected & available

        # Apply selected notes to the SAME frozen model.
        x = fuse(
            builder, meta, notes, anchors,
            used, "semantic_anchor"
        )

        after = probabilities(model, x)
        after_pred = after.argmax(axis=1)

        corrected = (
            errors & (after_pred == actual)
        )
        regressed = (
            ~errors & (after_pred != actual)
        )

        macro_f1 = f1_score(
            actual, after_pred,
            average="macro", zero_division=0
        )

        return {
            "strategy": name,
            "random_seed": seed,
            "prompt_opportunities": budget,
            "baseline_errors_targeted": int(
                (selected & errors).sum()
            ),
            "prompt_precision": float(
                errors[selected].mean()
            ),
            "notes_used": int(used.sum()),
            "effective_tfidf_notes": int(
                (used & tfidf_nonzero).sum()
            ),
            "wrong_to_correct": int(
                corrected.sum()
            ),
            "correct_to_wrong": int(
                regressed.sum()
            ),
            "net_corrections": int(
                corrected.sum() - regressed.sum()
            ),
            "macro_f1": float(macro_f1),
            "macro_f1_change_vs_no_notes": float(
                macro_f1 - no_note_f1
            ),
        }

    rows = [
        evaluate("ACTM", actm_mask)
    ]

    # Deterministic comparators.
    baselines = budget_matched_selection_baselines(
        before,
        conflicts,
        record_ids,
        prompt_budget=fraction,
        seed=42,
    )

    for name in (
        "Entropy-only",
        "Margin-only",
        "Conflict-only",
    ):
        rows.append(
            evaluate(name, baselines[name])
        )

    # Repeat random sampling for a more reliable
    # description of random-selection variability.
    for seed in range(42, 62):
        random_baselines = (
            budget_matched_selection_baselines(
                before,
                conflicts,
                record_ids,
                prompt_budget=fraction,
                seed=seed,
            )
        )

        rows.append(
            evaluate(
                "Random",
                random_baselines["Random"],
                seed=seed,
            )
        )

    result = pd.DataFrame(rows)

    output = (
        V2 / "outputs/diagnostics/"
        "actm_trigger_quality_seed_42"
    )
    output.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        output / "policy_comparison.csv",
        index=False,
    )

    columns = [
        "prompt_precision",
        "notes_used",
        "wrong_to_correct",
        "correct_to_wrong",
        "net_corrections",
        "macro_f1",
        "macro_f1_change_vs_no_notes",
    ]

    summary = (
        result.groupby("strategy")[columns]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        output / "policy_summary.csv"
    )

    print("\nNo-note Macro F1:", round(no_note_f1, 5))
    print("Matched prompt budget:", budget)
    print("\nStrategy summary:")
    print(summary.to_string())
    print("\nSaved to:", output)
    print("Held-out test evaluated: NO")


if __name__ == "__main__":
    main()
