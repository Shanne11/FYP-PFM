
"""
Tier A V2: Federated validation and Smart Note ablations.

Models:
    FedAvg
    FedProx
    Proposed

Proposed note strategies:
    none      - No Smart Notes
    selective - ACTM-selected Smart Notes
    always    - All available Smart Notes

Aggregation:
    Proposed default: bounded heuristic-adjusted FedAvg
    Proposed --disable-signal-adjustment: neutral FedAvg

Research safeguards:
    V1 remains unchanged.
    Uses frozen V2 transaction-ID group-safe split.
    Fits all preprocessing on training data only.
    Evaluates validation data only.
    Never scores the held-out test set.
    Does not overwrite existing result folders.
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

from scipy.sparse import hstack
from sklearn.metrics import log_loss, recall_score

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

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

from utils.federated_baseline import (
    _local_train,
    _sample_weighted_average,
)

from train_proposed import (
    set_seed,
    probabilities,
    local_train,
    note_mask,
    fuse,
    metric_row,
)


# ============================================================
# 1. Experiment naming
# ============================================================

def experiment_label(args):
    """Give each ablation a distinct output folder."""

    if args.model != "Proposed":
        return args.model.lower()

    if args.disable_signal_adjustment:
        if args.note_strategy == "selective":
            # Preserve the existing selective ablation name.
            return "proposed_no_signal_adjustment"

        return (
            f"proposed_notes_{args.note_strategy}"
            "_no_signal_adjustment"
        )

    return "proposed"


# ============================================================
# 2. Note statistics
# ============================================================

def note_statistics(frame, note_matrix, mask):
    """
    Count available and selected notes.

    A nonzero TF-IDF row means the note contains at least
    one recognised vocabulary feature.

    This does NOT prove that the note is semantically useful.
    """

    available = (
        frame["notes"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .to_numpy()
    )

    tfidf_nonzero = np.asarray(
        note_matrix.getnnz(axis=1)
    ).reshape(-1) > 0

    selected = np.asarray(mask, dtype=bool)

    return {
        "available_note_rows": int(available.sum()),
        "available_tfidf_note_rows": int(
            (available & tfidf_nonzero).sum()
        ),
        "selected_note_rows": int(selected.sum()),
        "selected_tfidf_note_rows": int(
            (selected & tfidf_nonzero).sum()
        ),
    }


# ============================================================
# 3. Main experiment
# ============================================================

def run(args):

    set_seed(args.seed)

    # --------------------------------------------------------
    # Load the frozen V2 dataset
    # --------------------------------------------------------

    train, val, test, manifest = load_experiment_data(
        str(V2 / "data/clean_budgetwise_v2.csv"),
        str(V2 / "data/experiment_split_v2.json"),
    )

    if manifest.get("group_key") != "transaction_id":
        raise ValueError(
            "Expected transaction-ID group-safe V2 manifest."
        )

    if (len(train), len(val), len(test)) != (
        9108, 2102, 2801
    ):
        raise ValueError("V2 split row counts have changed.")

    train_ids = set(train["transaction_id"])
    val_ids = set(val["transaction_id"])
    test_ids = set(test["transaction_id"])

    if (
        not train_ids.isdisjoint(val_ids)
        or not train_ids.isdisjoint(test_ids)
        or not val_ids.isdisjoint(test_ids)
    ):
        raise ValueError(
            "Transaction-ID overlap detected between splits."
        )

    print("V2 group-safe split verified.")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Held-out test: {len(test)} (NOT scored)")

    # --------------------------------------------------------
    # Training-only feature fitting
    # --------------------------------------------------------

    builder = V2FeatureBuilder().fit(train)

    if (
        builder.metadata_size != 71
        or builder.note_size != 37
        or len(builder.category_encoder.classes_) != 13
    ):
        raise ValueError("V2 feature contract mismatch.")

    feature_count = 108
    class_count = 13

    # --------------------------------------------------------
    # Class weights from training labels only
    # --------------------------------------------------------

    train_labels = builder.category_encoder.transform(
        train["category"]
    )

    weights = effective_number_weights(
        train_labels,
        classes=class_count,
        beta=0.9999,
    )

    # --------------------------------------------------------
    # Federated client partitions
    # --------------------------------------------------------

    client_ids = sorted(
        train["user_id"].astype(str).unique()
    )

    if len(client_ids) != 150:
        raise ValueError("Expected 150 V2 clients.")

    if args.max_clients is not None:
        client_ids = client_ids[:args.max_clients]

    client_data = {}

    for cid in client_ids:
        frame = train.loc[
            train["user_id"].astype(str) == cid
        ].copy()

        client_data[cid] = (
            frame,
            builder.transform_parts(frame),
        )

    # --------------------------------------------------------
    # Validation features
    # --------------------------------------------------------

    (
        val_meta,
        val_notes,
        val_anchors,
        val_y,
    ) = builder.transform_parts(val)

    # --------------------------------------------------------
    # ACTM and training-fitted context detector
    # --------------------------------------------------------

    detector = CrossAccountConflictDetector().fit(
        train
    )

    val_conflicts = detector.transform(val)

    actm = ACTM(
        ACTMConfig(
            0.65,  # Normalised entropy threshold
            0.15,  # Top-two margin threshold
            0.30,  # Maximum prompt fraction
        )
    )

    # --------------------------------------------------------
    # Global model
    # --------------------------------------------------------

    model = MLP(
        feature_count,
        class_count,
    )

    # ========================================================
    # 4. Output configuration
    # ========================================================

    label = experiment_label(args)

    stage = (
        "smoke"
        if args.max_clients is not None
        else "full"
    )

    output = (
        V2
        / "outputs"
        / "validation"
        / stage
        / f"{label}_seed_{args.seed}"
    )

    # Prevent accidental overwriting of existing research runs.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {output}\n"
            "Preserve the existing results and choose a "
            "different experiment configuration or folder."
        )

    output.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        builder,
        output / "feature_pipeline_v2.joblib",
    )

    print("\nExperiment:", label)
    print("Model:", args.model)
    print("Note strategy:", args.note_strategy)
    print(
        "Signal adjustment enabled:",
        (
            args.model == "Proposed"
            and not args.disable_signal_adjustment
        ),
    )
    print("Clients:", len(client_ids))
    print("Rounds:", args.rounds)
    print("Local epochs:", args.local_epochs)
    print("Output:", output)

    # ========================================================
    # 5. Federated training
    # ========================================================

    rows = []
    best_score = -1.0
    best_round = None
    best_metrics = None

    for round_no in range(1, args.rounds + 1):

        updates = []

        train_selected_notes = 0
        train_effective_notes = 0
        train_actm_prompts = 0

        # FedProx reference: global parameters before
        # this communication round's client updates.
        global_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        # ----------------------------------------------------
        # Train each client
        # ----------------------------------------------------

        for cid in client_ids:

            frame, parts = client_data[cid]
            meta, notes, anchors, y = parts

            local_model = copy.deepcopy(model)

            # ================================================
            # Proposed method and note ablations
            # ================================================

            if args.model == "Proposed":

                # Metadata-only prediction before clarification.
                before = probabilities(
                    model,
                    builder.metadata_only(meta),
                )

                # ACTM is used for the selective strategy.
                # None and always do not request ACTM prompts.
                if args.note_strategy == "selective":
                    decisions = actm.decide(
                        before,
                        detector.transform(frame),
                    )
                    train_actm_prompts += int(
                        decisions["triggered"].sum()
                    )
                else:
                    decisions = None

                # Change ONLY which notes are available:
                # none, selective or always.
                mask = note_mask(
                    frame,
                    decisions,
                    args.note_strategy,
                )

                stats = note_statistics(
                    frame,
                    notes,
                    mask,
                )

                train_selected_notes += (
                    stats["selected_note_rows"]
                )

                train_effective_notes += (
                    stats["selected_tfidf_note_rows"]
                )

                # All strategies use the same semantic-anchor
                # fusion method and the same 108-input MLP.
                x = fuse(
                    builder,
                    meta,
                    notes,
                    anchors,
                    mask,
                    "semantic_anchor",
                )

                # Local optimisation is identical across
                # the three note strategies.
                loss = local_train(
                    local_model,
                    x,
                    y,
                    args.local_epochs,
                    0.001,
                    weights,
                )

                # Clarification-derived heuristic.
                #
                # In no-signal experiments this is not used
                # to adjust the aggregation coefficients.
                after = probabilities(
                    local_model,
                    x,
                )

                selected = np.flatnonzero(mask)
                mean_heuristic = None

                if len(selected):

                    (
                        score,
                        _,
                        _,
                        _,
                        nonempty,
                    ) = clarification_heuristic(
                        before[selected],
                        after[selected],
                        notes[selected],
                        anchors[selected],
                        frame.iloc[selected][
                            "notes"
                        ].tolist(),
                        weights=(0.5, 0.3, 0.2),
                    )

                    valid = score[nonempty]

                    if len(valid):
                        mean_heuristic = float(
                            valid.mean()
                        )

                updates.append({
                    "client_id": cid,
                    "weights": copy.deepcopy(
                        local_model.state_dict()
                    ),
                    "sample_count": len(frame),
                    "note_count": int(mask.sum()),
                    "mean_clarification_heuristic": (
                        mean_heuristic
                    ),
                    "local_loss": loss,
                })

            # ================================================
            # FedAvg and FedProx
            # ================================================

            else:

                # Historical V2 baselines use all available
                # note features via simple concatenation.
                x = hstack(
                    [meta, notes],
                    format="csr",
                )

                mu = (
                    0.01
                    if args.model == "FedProx"
                    else 0.0
                )

                state, loss = _local_train(
                    local_model,
                    x,
                    y,
                    global_parameters,
                    mu,
                    args.local_epochs,
                    0.001,
                    weights,
                )

                updates.append({
                    "client_id": cid,
                    "weights": state,
                    "sample_count": len(frame),
                    "local_loss": loss,
                })

        # ====================================================
        # 6. Server aggregation
        # ====================================================

        if args.model == "Proposed":

            if args.disable_signal_adjustment:

                # Neutralise the clarification heuristic.
                # Retain Proposed's update validation checks.
                # All valid clients receive multiplier 1.0.
                neutral_updates = [
                    {
                        **update,
                        "mean_clarification_heuristic": None,
                        "note_count": 0,
                    }
                    for update in updates
                ]

                averaged, diagnostics = (
                    bounded_signal_adjusted_average(
                        neutral_updates,
                        min_notes=1,
                    )
                )

            else:

                # Full Proposed method:
                # bounded heuristic-adjusted FedAvg.
                averaged, diagnostics = (
                    bounded_signal_adjusted_average(
                        updates,
                        min_notes=1,
                    )
                )

            if averaged is None:
                raise RuntimeError(
                    "No valid client updates."
                )

        else:

            averaged = _sample_weighted_average(
                updates
            )

        model.load_state_dict(averaged)

        # ====================================================
        # 7. Validation evaluation
        # ====================================================

        if args.model == "Proposed":

            if args.note_strategy == "selective":

                val_before = probabilities(
                    model,
                    builder.metadata_only(val_meta),
                )

                val_decisions = actm.decide(
                    val_before,
                    val_conflicts,
                )

                prompts = int(
                    val_decisions["triggered"].sum()
                )

            else:

                # No ACTM prompting for none or always.
                val_decisions = None
                prompts = 0

            val_mask = note_mask(
                val,
                val_decisions,
                args.note_strategy,
            )

            val_note_stats = note_statistics(
                val,
                val_notes,
                val_mask,
            )

            val_x = fuse(
                builder,
                val_meta,
                val_notes,
                val_anchors,
                val_mask,
                "semantic_anchor",
            )

        else:

            val_x = hstack(
                [val_meta, val_notes],
                format="csr",
            )

            prompts = 0

            # Baseline input contains the note feature block.
            available = (
                val["notes"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                .to_numpy()
            )

            val_note_stats = note_statistics(
                val,
                val_notes,
                available,
            )

        # ----------------------------------------------------
        # Predict validation labels
        # ----------------------------------------------------

        prob = probabilities(
            model,
            val_x,
        )

        predicted = prob.argmax(axis=1)

        metrics = metric_row(
            val_y,
            prob,
        )

        recalls = recall_score(
            val_y,
            predicted,
            labels=np.arange(class_count),
            average=None,
            zero_division=0,
        )

        result = {
            "round": round_no,

            "validation_loss": float(
                log_loss(
                    val_y,
                    prob,
                    labels=np.arange(class_count),
                )
            ),

            "predicted_categories": int(
                np.unique(predicted).size
            ),

            "zero_recall_categories": int(
                (recalls == 0).sum()
            ),

            # ACTM prompts only, not notes already available.
            "validation_prompts": prompts,

            # Actual rows whose notes were selected.
            "validation_available_notes": (
                val_note_stats["available_note_rows"]
            ),

            "validation_selected_notes": (
                val_note_stats["selected_note_rows"]
            ),

            # Selected notes with nonzero TF-IDF features.
            "validation_effective_tfidf_notes": (
                val_note_stats["selected_tfidf_note_rows"]
            ),

            "training_selected_notes": (
                train_selected_notes
            ),

            "training_effective_tfidf_notes": (
                train_effective_notes
            ),

            "training_actm_prompts": (
                train_actm_prompts
            ),

            **metrics,
        }

        rows.append(result)

        # Save a row after every round.
        pd.DataFrame(rows).to_csv(
            output / "round_metrics.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Select best checkpoint by validation Macro F1
        # ----------------------------------------------------

        if result["macro_f1"] > best_score:

            best_score = result["macro_f1"]
            best_round = round_no
            best_metrics = result.copy()

            torch.save(
                model.state_dict(),
                output / "best_model.pt",
            )

        print(
            f"{label} | "
            f"round {round_no}/{args.rounds} | "
            f"best F1={best_score:.5f} | "
            f"coverage={result['predicted_categories']}/13 | "
            f"notes={result['validation_selected_notes']} | "
            f"prompts={prompts}",
            flush=True,
        )

    # ========================================================
    # 8. Save experiment summary
    # ========================================================

    signal_enabled = (
        args.model == "Proposed"
        and not args.disable_signal_adjustment
    )

    aggregation = (
        "bounded_heuristic_adjusted_fedavg"
        if signal_enabled
        else "sample_weighted_fedavg"
    )

    summary = {
        "version": "V2",
        "experiment_variant": label,
        "model": args.model,
        "seed": args.seed,

        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows_scored": 0,

        "clients": len(client_ids),
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "learning_rate": 0.001,

        "features": feature_count,
        "classes": class_count,

        "loss": "effective_number_beta_0.9999",

        "actm_enabled": (
            args.model == "Proposed"
            and args.note_strategy == "selective"
        ),

        "note_strategy": (
            args.note_strategy
            if args.model == "Proposed"
            else "all_available"
        ),

        "fusion_mode": (
            "semantic_anchor"
            if args.model == "Proposed"
            else "simple_concat"
        ),

        "signal_adjustment_enabled": signal_enabled,
        "aggregation": aggregation,

        "fedprox_mu": (
            0.01 if args.model == "FedProx" else 0.0
        ),

        "split_version": manifest["version"],
        "stage": stage,

        "best_round": best_round,
        "best_validation": best_metrics,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment completed.")
    print("Variant:", label)
    print("Best round:", best_round)
    print(f"Best Macro F1: {best_score:.5f}")
    print("Test set evaluated: NO")
    print("Results:", output)


# ============================================================
# 9. Command-line arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="V2 validation-only federated experiments"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "FedAvg",
            "FedProx",
            "Proposed",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
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

    parser.add_argument(
        "--note-strategy",
        choices=[
            "none",
            "selective",
            "always",
        ],
        default="selective",
        help="Smart Note strategy for Proposed",
    )

    parser.add_argument(
        "--disable-signal-adjustment",
        action="store_true",
        help=(
            "Use neutral sample-weighted aggregation "
            "instead of heuristic-adjusted aggregation."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Prevent invalid experimental configurations
    # --------------------------------------------------------

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

    if args.model != "Proposed":

        if args.note_strategy != "selective":
            parser.error(
                "--note-strategy is only supported "
                "for Proposed."
            )

        if args.disable_signal_adjustment:
            parser.error(
                "--disable-signal-adjustment is only "
                "supported for Proposed."
            )

    # The Smart Note ablation must keep aggregation fixed.
    # Therefore, none/always are required to use FedAvg.
    if (
        args.model == "Proposed"
        and args.note_strategy != "selective"
        and not args.disable_signal_adjustment
    ):
        parser.error(
            "For Smart Note ablations, use "
            "--disable-signal-adjustment to keep "
            "aggregation fixed across strategies."
        )

    run(args)
