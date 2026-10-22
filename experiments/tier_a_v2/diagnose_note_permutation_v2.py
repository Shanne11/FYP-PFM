
"""
Tier A V2 - Smart Note Permutation Diagnostic

Compare:
1. No notes
2. Original ACTM-selected notes
3. Shuffled ACTM-selected notes (50 permutations)

Uses one frozen validation-selected model.
No retraining and no held-out test evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Needed when loading the saved feature builder.
from prepare_features_v2 import V2FeatureBuilder

from models.mlp import MLP
from models.federated.actm import (
    ACTM, ACTMConfig, CrossAccountConflictDetector
)
from utils.experiment_data import load_experiment_data
from train_proposed import probabilities, fuse


def evaluate(model, builder, meta, notes, anchors,
             mask, actual, before_pred):

    x = fuse(
        builder, meta, notes, anchors,
        mask, "semantic_anchor"
    )

    prob = probabilities(model, x)
    pred = prob.argmax(axis=1)

    corrected = (
        (before_pred != actual) & (pred == actual)
    )

    regressed = (
        (before_pred == actual) & (pred != actual)
    )

    n_corrected = int(corrected.sum())
    n_regressed = int(regressed.sum())

    return {
        "macro_f1": float(f1_score(
            actual, pred, average="macro", zero_division=0
        )),
        "accuracy": float(accuracy_score(actual, pred)),
        "wrong_to_correct": n_corrected,
        "correct_to_wrong": n_regressed,
        "net_corrections": n_corrected - n_regressed,
    }


def main(args):

    train, val, test, manifest = load_experiment_data(
        str(V2 / "data/clean_budgetwise_v2.csv"),
        str(V2 / "data/experiment_split_v2.json"),
    )

    assert manifest["group_key"] == "transaction_id"
    assert len(val) == 2102
    assert len(test) == 2801

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
        builder.transform_parts(val)
    )

    before = probabilities(
        model, builder.metadata_only(meta)
    )

    before_pred = before.argmax(axis=1)

    detector = CrossAccountConflictDetector().fit(train)

    actm = ACTM(ACTMConfig(0.65, 0.15, 0.30))

    decisions = actm.decide(
        before, detector.transform(val)
    )

    prompted = decisions["triggered"].to_numpy()

    available = (
        val["notes"].fillna("").astype(str)
        .str.strip().ne("").to_numpy()
    )

    used = prompted & available
    positions = np.flatnonzero(used)

    if not len(positions):
        raise RuntimeError("No selected notes available")

    # The number of ACTM-selected notes is fixed.
    # Only their assignment to selected transactions changes.

    no_mask = np.zeros(len(val), dtype=bool)

    baseline = evaluate(
        model, builder, meta, notes, anchors,
        no_mask, actual, before_pred
    )

    original = evaluate(
        model, builder, meta, notes, anchors,
        used, actual, before_pred
    )

    rng = np.random.default_rng(args.seed)
    rows = []

    for iteration in range(args.permutations):

        index_map = np.arange(len(val))

        # Shuffle only among transactions that actually
        # received notes under ACTM.
        # This preserves the selected-note distribution.
        index_map[positions] = rng.permutation(positions)

        shuffled_notes = notes[index_map]

        result = evaluate(
            model, builder, meta, shuffled_notes,
            anchors, used, actual, before_pred
        )

        rows.append({
            "permutation": iteration + 1,
            **result,
        })

        print(
            f"Permutation {iteration + 1}/"
            f"{args.permutations}: "
            f"Macro F1={result['macro_f1']:.5f}",
            flush=True,
        )

    table = pd.DataFrame(rows)

    output = (
        V2 / "outputs/diagnostics/"
        f"note_permutation_seed_{args.seed}"
    )

    output.mkdir(parents=True, exist_ok=True)

    table.to_csv(
        output / "permutation_results.csv",
        index=False
    )

    summary = {
        "experiment": "V2 note alignment permutation",
        "partition": "validation",
        "checkpoint": checkpoint.name,
        "seed": args.seed,
        "permutations": args.permutations,
        "validation_rows": len(val),
        "test_rows_scored": 0,
        "actm_prompts": int(prompted.sum()),
        "notes_used": int(used.sum()),
        "no_notes": baseline,
        "original_notes": original,
        "shuffled_notes": {
            "mean_macro_f1": float(
                table["macro_f1"].mean()
            ),
            "std_macro_f1": float(
                table["macro_f1"].std(ddof=1)
            ),
            "min_macro_f1": float(
                table["macro_f1"].min()
            ),
            "max_macro_f1": float(
                table["macro_f1"].max()
            ),
            "mean_net_corrections": float(
                table["net_corrections"].mean()
            ),
            "fraction_shuffled_f1_ge_original": float(
                (table["macro_f1"] >= original["macro_f1"]).mean()
            ),
        },
        "limitations": (
            "Inference-only control with a frozen model. "
            "Does not establish causal effects of training "
            "with different note policies."
        ),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--permutations", type=int, default=50
    )

    args = parser.parse_args()

    if args.permutations < 2:
        parser.error("--permutations must be at least 2")

    main(args)
