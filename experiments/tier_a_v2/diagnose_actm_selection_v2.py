
"""V2 validation-only ACTM selection diagnostic."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prepare_features_v2 import V2FeatureBuilder
from models.mlp import MLP
from models.federated.actm import (
    ACTM, ACTMConfig, CrossAccountConflictDetector
)
from utils.experiment_data import load_experiment_data
from train_proposed import probabilities, fuse, metric_row


def main():
    train, validation, _, manifest = load_experiment_data(
        str(V2 / "data/clean_budgetwise_v2.csv"),
        str(V2 / "data/experiment_split_v2.json"),
    )

    assert manifest["group_key"] == "transaction_id"

    # Existing selective-notes model without heuristic weighting.
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

    metadata, notes, anchors, actual = (
        builder.transform_parts(validation)
    )

    # Predictions without notes.
    before = probabilities(
        model, builder.metadata_only(metadata)
    )
    before_prediction = before.argmax(axis=1)

    # Compute ACTM decisions using training-fitted context.
    detector = CrossAccountConflictDetector().fit(train)
    conflicts = detector.transform(validation)

    actm = ACTM(ACTMConfig(0.65, 0.15, 0.30))
    decisions = actm.decide(before, conflicts)

    actm_selected = decisions["triggered"].to_numpy()
    budget = int(actm_selected.sum())

    # Random comparator receives exactly the same
    # number of prompt opportunities.
    rng = np.random.default_rng(42)
    random_selected = np.zeros(
        len(validation), dtype=bool
    )
    random_positions = rng.choice(
        len(validation),
        size=budget,
        replace=False,
    )
    random_selected[random_positions] = True

    # Retrospective notes are available in the dataset.
    available_notes = (
        validation["notes"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .to_numpy()
    )

    note_has_tfidf = (
        np.asarray(notes.getnnz(axis=1)).reshape(-1) > 0
    )

    selections = {
        "No notes": np.zeros(len(validation), dtype=bool),
        "ACTM selective": actm_selected,
        "Random matched budget": random_selected,
        "All available notes": available_notes,
    }

    results = []

    for name, selected in selections.items():
        used = selected & available_notes

        x = fuse(
            builder,
            metadata,
            notes,
            anchors,
            used,
            "semantic_anchor",
        )

        after = probabilities(model, x)
        predicted = after.argmax(axis=1)

        metrics = metric_row(actual, after)

        # Did the existing model change a wrong prediction
        # into a correct one when using selected notes?
        wrong_to_correct = int(np.sum(
            (before_prediction != actual)
            & (predicted == actual)
        ))

        correct_to_wrong = int(np.sum(
            (before_prediction == actual)
            & (predicted != actual)
        ))

        row = {
            "strategy": name,
            "prompt_opportunities": (
                int(selected.sum())
                if name in [
                    "ACTM selective", "Random matched budget"
                ]
                else 0
            ),
            "notes_used": int(used.sum()),
            "notes_with_tfidf": int(
                (used & note_has_tfidf).sum()
            ),
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_corrections": (
                wrong_to_correct - correct_to_wrong
            ),
            **metrics,
        }

        results.append(row)

    output = (
        V2 / "outputs/diagnostics/"
        "actm_selection_seed_42"
    )
    output.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(results)
    table.to_csv(
        output / "selection_comparison.csv",
        index=False,
    )

    summary = {
        "experiment": "ACTM selection diagnostic",
        "checkpoint": str(checkpoint),
        "data_partition": "validation",
        "validation_rows": len(validation),
        "actm_prompt_budget": budget,
        "test_rows_scored": 0,
        "comparison_type": "inference_only",
        "important_limitation": (
            "All selection policies are evaluated using "
            "one frozen model trained with selective notes. "
            "This does not replace separately retrained "
            "policy comparisons."
        ),
        "results": results,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(table.to_string(index=False))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
