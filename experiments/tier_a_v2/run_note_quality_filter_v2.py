
"""
Tier A V2 - Isolated Smart Note Quality Filtering Experiment

Compares selective Smart Notes with and without an objective
low-information note filter.

Preserves:
- Original V2 dataset and frozen group-safe split
- Original 108-feature training-fitted representation
- ACTM thresholds and prompt budget
- Semantic-anchor feature fusion
- Clarification-derived heuristic
- Bounded heuristic-adjusted FedAvg
- Effective-number loss and federated configuration

Does not:
- Change V1
- Rewrite source categories
- Use category labels to decide whether notes are good
- Fit preprocessing on validation/test data
- Score the held-out test partition
- Overwrite existing experiment folders
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import log_loss, recall_score

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2))

from prepare_features_v2 import V2FeatureBuilder
from models.mlp import MLP
from models.federated.actm import (
    ACTM,
    ACTMConfig,
    CrossAccountConflictDetector,
)
from models.federated.heuristic import (
    clarification_heuristic,
    bounded_signal_adjusted_average,
)
from utils.class_balance import effective_number_weights
from utils.experiment_data import load_experiment_data
from train_proposed import (
    set_seed,
    probabilities,
    local_train,
    fuse,
    metric_row,
)


# Matches the objective low-information list in
# experiments/tier_a_v2/build_dataset_v2.py.
#
# Do not add category-based rules such as:
#   "monthly rent payment" must have category Rent.
#
# That would use label semantics to construct evaluation
# inputs and change the research question.
LOW_INFORMATION_NOTES = frozenset({
    "",
    "n/a",
    "unknown",
    "asdfgh",
    "xyz123",
    "!!!",
    "...",
    "misc",
    "test",
    "null",
})


def note_quality_masks(frame):
    """
    Return:
      available: nonempty source note
      low_info:  exact case-insensitive placeholder match

    No transaction categories are inspected here.
    """

    text = (
        frame["notes"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    available = text.ne("").to_numpy()

    low_info = (
        text.isin(LOW_INFORMATION_NOTES)
        .to_numpy()
    )

    return available, low_info


def selected_note_mask(frame, decisions, filter_mode):
    """
    ACTM prompt selection stays unchanged.

    Only the availability of notes AFTER an ACTM prompt
    is affected by the quality filter.
    """

    available, low_info = note_quality_masks(frame)

    prompted = decisions["triggered"].to_numpy(
        dtype=bool
    )

    # Original V2 selection before filtering.
    original_mask = prompted & available

    if filter_mode == "placeholder":
        filtered_mask = original_mask & ~low_info

    elif filter_mode == "off":
        filtered_mask = original_mask.copy()

    else:
        raise ValueError(
            f"Unknown filter mode: {filter_mode}"
        )

    return original_mask, filtered_mask


def note_counts(frame):
    available, low_info = note_quality_masks(frame)

    return {
        "rows": int(len(frame)),
        "available_notes": int(available.sum()),
        "low_information_notes": int(
            (available & low_info).sum()
        ),
        "empty_notes": int((~available).sum()),
        "eligible_after_filter": int(
            (available & ~low_info).sum()
        ),
    }


def run(args):

    set_seed(args.seed)

    # ========================================================
    # 1. Load and validate the existing V2 split
    # ========================================================

    train, validation, test, manifest = (
        load_experiment_data(
            str(V2 / "data/clean_budgetwise_v2.csv"),
            str(V2 / "data/experiment_split_v2.json"),
        )
    )

    if manifest.get("group_key") != "transaction_id":
        raise ValueError(
            "Expected transaction-ID group-safe split."
        )

    if (len(train), len(validation), len(test)) != (
        9108, 2102, 2801
    ):
        raise ValueError(
            "Unexpected V2 split sizes."
        )

    groups = [
        set(frame["transaction_id"].astype(str))
        for frame in (train, validation, test)
    ]

    if (
        not groups[0].isdisjoint(groups[1])
        or not groups[0].isdisjoint(groups[2])
        or not groups[1].isdisjoint(groups[2])
    ):
        raise ValueError(
            "Transaction-ID leakage between partitions."
        )

    train_quality = note_counts(train)
    validation_quality = note_counts(validation)

    # ========================================================
    # 2. Preserve the ORIGINAL feature representation
    # ========================================================

    # Critical:
    # Fit on the unchanged training data.
    #
    # Filtering is applied later to the Smart Note mask.
    # This keeps the vocabulary and 108 feature positions
    # identical to the earlier V2 experiments.

    builder = V2FeatureBuilder().fit(train)

    if (
        builder.metadata_size != 71
        or builder.note_size != 37
        or len(builder.category_encoder.classes_) != 13
    ):
        raise ValueError(
            "V2 108-feature contract mismatch."
        )

    feature_count = 108
    class_count = 13

    y_train = builder.category_encoder.transform(
        train["category"]
    )

    class_weights = effective_number_weights(
        y_train,
        classes=class_count,
        beta=0.9999,
    )

    # ========================================================
    # 3. Prepare client partitions
    # ========================================================

    clients = sorted(
        train["user_id"].astype(str).unique()
    )

    if len(clients) != 150:
        raise ValueError(
            "Expected 150 simulated clients."
        )

    if args.max_clients is not None:
        clients = clients[:args.max_clients]

    client_data = {}

    for client_id in clients:

        frame = train.loc[
            train["user_id"].astype(str) == client_id
        ].copy()

        client_data[client_id] = (
            frame,
            builder.transform_parts(frame),
        )

    # ========================================================
    # 4. Validation features
    # ========================================================

    (
        val_meta,
        val_notes,
        val_anchors,
        val_y,
    ) = builder.transform_parts(validation)

    # Fitted using training data only.
    detector = CrossAccountConflictDetector().fit(
        train
    )

    val_conflicts = detector.transform(validation)

    actm = ACTM(
        ACTMConfig(
            entropy_threshold=0.65,
            margin_threshold=0.15,
            prompt_budget=0.30,
        )
    )

    model = MLP(feature_count, class_count)

    # ========================================================
    # 5. Isolated experiment output
    # ========================================================

    stage = (
        "smoke"
        if args.max_clients is not None
        else "full"
    )

    variant = (
        f"proposed_note_quality_{args.filter_mode}_"
        f"{args.aggregation}_seed_{args.seed}"
    )

    output = (
        V2 / "outputs"
        / "validation"
        / stage
        / variant
    )

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Existing results would be overwritten: {output}"
        )

    output.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        builder,
        output / "feature_pipeline_v2.joblib",
    )

    # ========================================================
    # 6. Federated training
    # ========================================================

    history = []
    client_diagnostics = []

    best_f1 = -1.0
    best_round = None
    best_metrics = None

    print("\n=== V2 NOTE QUALITY EXPERIMENT ===")
    print("Filter:", args.filter_mode)
    print("Aggregation:", args.aggregation)
    print("Clients:", len(clients))
    print("Rounds:", args.rounds)
    print("Local epochs:", args.local_epochs)
    print("Features:", feature_count)
    print("Training note quality:", train_quality)
    print("Validation note quality:", validation_quality)
    print("Held-out test evaluated: NO")
    print("Output:", output)

    for round_no in range(1, args.rounds + 1):

        updates = []

        train_prompts = 0
        train_notes_before_filter = 0
        train_notes_after_filter = 0
        train_effective_notes = 0

        # ----------------------------------------------------
        # 6A. Client-side training
        # ----------------------------------------------------

        for client_id in clients:

            frame, parts = client_data[client_id]

            meta, notes, anchors, y = parts

            local_model = copy.deepcopy(model)

            # Metadata-only prediction before clarification.
            before = probabilities(
                model,
                builder.metadata_only(meta),
            )

            # Unchanged ACTM selection.
            decisions = actm.decide(
                before,
                detector.transform(frame),
            )

            train_prompts += int(
                decisions["triggered"].sum()
            )

            original_mask, selected_mask = (
                selected_note_mask(
                    frame,
                    decisions,
                    args.filter_mode,
                )
            )

            train_notes_before_filter += int(
                original_mask.sum()
            )

            train_notes_after_filter += int(
                selected_mask.sum()
            )

            tfidf_nonzero = (
                np.asarray(
                    notes.getnnz(axis=1)
                ).reshape(-1) > 0
            )

            train_effective_notes += int(
                (selected_mask & tfidf_nonzero).sum()
            )

            # Same semantic-anchor fusion.
            # The only difference is the note selection mask.
            x = fuse(
                builder,
                meta,
                notes,
                anchors,
                selected_mask,
                "semantic_anchor",
            )

            # Same local training procedure.
            local_loss = local_train(
                local_model,
                x,
                y,
                args.local_epochs,
                0.001,
                class_weights,
            )

            # Clarification-derived heuristic.
            # Preserve the original implementation.
            after = probabilities(
                local_model,
                x,
            )

            selected_positions = np.flatnonzero(
                selected_mask
            )

            mean_heuristic = None

            if len(selected_positions) > 0:

                (
                    scores,
                    _,
                    _,
                    _,
                    nonempty,
                ) = clarification_heuristic(
                    before[selected_positions],
                    after[selected_positions],
                    notes[selected_positions],
                    anchors[selected_positions],
                    frame.iloc[selected_positions][
                        "notes"
                    ].tolist(),
                    weights=(0.5, 0.3, 0.2),
                )

                valid_scores = scores[nonempty]

                if len(valid_scores):
                    mean_heuristic = float(
                        valid_scores.mean()
                    )

            updates.append({
                "client_id": client_id,
                "weights": copy.deepcopy(
                    local_model.state_dict()
                ),
                "sample_count": len(frame),
                "note_count": int(
                    selected_mask.sum()
                ),
                "mean_clarification_heuristic": (
                    mean_heuristic
                ),
                "local_loss": local_loss,
            })

        # ----------------------------------------------------
        # 6B. Federated aggregation
        # ----------------------------------------------------

        if args.aggregation == "neutral":

            # All valid client multipliers become 1.0.
            # Keeps the original aggregation validity checks.
            aggregation_updates = [
                {
                    **update,
                    "mean_clarification_heuristic": None,
                    "note_count": 0,
                }
                for update in updates
            ]

        else:
            aggregation_updates = updates

        averaged, aggregation_rows = (
            bounded_signal_adjusted_average(
                aggregation_updates,
                min_notes=1,
            )
        )

        if averaged is None:
            raise RuntimeError(
                "All federated client updates were rejected."
            )

        model.load_state_dict(averaged)

        # Save client multiplier diagnostics.
        for item in aggregation_rows:

            client_diagnostics.append({
                "round": round_no,
                **item,
            })

        # ----------------------------------------------------
        # 6C. Validation-only ACTM
        # ----------------------------------------------------

        val_before = probabilities(
            model,
            builder.metadata_only(val_meta),
        )

        val_decisions = actm.decide(
            val_before,
            val_conflicts,
        )

        val_prompts = int(
            val_decisions["triggered"].sum()
        )

        val_raw_mask, val_selected_mask = (
            selected_note_mask(
                validation,
                val_decisions,
                args.filter_mode,
            )
        )

        val_tfidf_nonzero = (
            np.asarray(
                val_notes.getnnz(axis=1)
            ).reshape(-1) > 0
        )

        val_x = fuse(
            builder,
            val_meta,
            val_notes,
            val_anchors,
            val_selected_mask,
            "semantic_anchor",
        )

        val_prob = probabilities(
            model,
            val_x,
        )

        val_pred = val_prob.argmax(axis=1)

        metrics = metric_row(
            val_y,
            val_prob,
        )

        recalls = recall_score(
            val_y,
            val_pred,
            labels=np.arange(class_count),
            average=None,
            zero_division=0,
        )

        result = {
            "round": round_no,

            "validation_loss": float(
                log_loss(
                    val_y,
                    val_prob,
                    labels=np.arange(class_count),
                )
            ),

            "predicted_categories": int(
                np.unique(val_pred).size
            ),

            "zero_recall_categories": int(
                (recalls == 0).sum()
            ),

            "validation_prompts": val_prompts,

            "validation_notes_before_filter": int(
                val_raw_mask.sum()
            ),

            "validation_notes_used": int(
                val_selected_mask.sum()
            ),

            "validation_notes_filtered": int(
                val_raw_mask.sum()
                - val_selected_mask.sum()
            ),

            "validation_effective_tfidf_notes": int(
                (
                    val_selected_mask
                    & val_tfidf_nonzero
                ).sum()
            ),

            "training_prompts": train_prompts,

            "training_notes_before_filter": (
                train_notes_before_filter
            ),

            "training_notes_used": (
                train_notes_after_filter
            ),

            "training_notes_filtered": (
                train_notes_before_filter
                - train_notes_after_filter
            ),

            "training_effective_tfidf_notes": (
                train_effective_notes
            ),

            **metrics,
        }

        history.append(result)

        pd.DataFrame(history).to_csv(
            output / "round_metrics.csv",
            index=False,
        )

        pd.DataFrame(client_diagnostics).to_csv(
            output / "client_aggregation_diagnostics.csv",
            index=False,
        )

        # ----------------------------------------------------
        # 6D. Best checkpoint selection
        # ----------------------------------------------------

        if result["macro_f1"] > best_f1:

            best_f1 = result["macro_f1"]
            best_round = round_no
            best_metrics = result.copy()

            torch.save(
                model.state_dict(),
                output / "best_model.pt",
            )

        print(
            f"Round {round_no}/{args.rounds} | "
            f"Macro F1={result['macro_f1']:.5f} | "
            f"Best={best_f1:.5f} | "
            f"Categories={result['predicted_categories']}/13 | "
            f"Notes used={result['validation_notes_used']} | "
            f"Filtered={result['validation_notes_filtered']}",
            flush=True,
        )

    # ========================================================
    # 7. Experiment summary
    # ========================================================

    summary = {
        "version": "V2",

        "experiment_variant": variant,

        "model": "Proposed",

        "note_filter": args.filter_mode,

        "note_filter_rule": sorted(
            LOW_INFORMATION_NOTES
        ),

        "filter_uses_category_labels": False,

        "aggregation": (
            "bounded_heuristic_adjusted_fedavg"
            if args.aggregation == "bounded"
            else "sample_weighted_fedavg"
        ),

        "signal_adjustment_enabled": (
            args.aggregation == "bounded"
        ),

        "actm_enabled": True,
        "note_strategy": "selective",
        "fusion_mode": "semantic_anchor",

        "seed": args.seed,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "clients": len(clients),

        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows_scored": 0,

        "split_version": manifest["version"],
        "dataset_fingerprint": manifest[
            "dataset_fingerprint"
        ],

        "features": feature_count,
        "metadata_features": builder.metadata_size,
        "note_features": builder.note_size,
        "classes": class_count,

        "loss": "effective_number_beta_0.9999",

        "feature_fitting": "original training rows only",

        "train_note_quality": train_quality,
        "validation_note_quality": validation_quality,

        "best_round": best_round,
        "best_validation": best_metrics,

        "limitations": [
            "Filtering removes only predefined placeholders.",
            "Semantically mismatched non-placeholder notes remain.",
            "Category labels are not used for filtering.",
            "Retrospective notes are not user-entered clarifications.",
            "Validation-only experiment; test set untouched.",
        ],
    }

    # --------------------------------------------------------
    # 8. Compare against the existing unfiltered result
    # --------------------------------------------------------

    existing_name = (
        "proposed_seed_42"
        if args.aggregation == "bounded"
        else "proposed_no_signal_adjustment_seed_42"
    )

    original_path = (
        V2 / "outputs/validation/full"
        / existing_name
        / "summary.json"
    )

    if (
        args.filter_mode == "placeholder"
        and args.seed == 42
        and args.max_clients is None
        and args.rounds == 40
        and args.local_epochs == 3
        and original_path.exists()
    ):

        original_summary = json.loads(
            original_path.read_text(encoding="utf-8")
        )

        original_f1 = float(
            original_summary[
                "best_validation"
            ]["macro_f1"]
        )

        filtered_f1 = float(
            best_metrics["macro_f1"]
        )

        summary["unfiltered_comparison"] = {
            "reference_experiment": existing_name,
            "unfiltered_macro_f1": original_f1,
            "filtered_macro_f1": filtered_f1,
            "macro_f1_difference": (
                filtered_f1 - original_f1
            ),
        }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== EXPERIMENT COMPLETE ===")
    print("Variant:", variant)
    print("Best round:", best_round)
    print("Best validation Macro F1:", best_f1)
    print("Held-out test scored: 0")

    if "unfiltered_comparison" in summary:
        print(
            "Difference vs original:",
            summary[
                "unfiltered_comparison"
            ]["macro_f1_difference"],
        )

    print("Saved:", output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filter-mode",
        choices=["off", "placeholder"],
        default="placeholder",
    )

    parser.add_argument(
        "--aggregation",
        choices=["bounded", "neutral"],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-clients",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.rounds < 1 or args.local_epochs < 1:
        parser.error(
            "Rounds and local epochs must be positive."
        )

    if (
        args.max_clients is not None
        and not 1 <= args.max_clients <= 150
    ):
        parser.error(
            "--max-clients must be between 1 and 150."
        )

    run(args)
