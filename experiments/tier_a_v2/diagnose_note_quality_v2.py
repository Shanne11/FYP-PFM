
"""
Tier A V2 - Smart Note Quality and Prediction Diagnostic

Purpose:
1. Identify categories affected by Smart Notes.
2. Find wrong-to-correct and correct-to-wrong changes.
3. Detect confidence increases on incorrect predictions.
4. Examine repeated source notes and category ambiguity.
5. Calculate a same-model clarification heuristic diagnostic.

Important:
- Validation only.
- Uses one frozen Proposed checkpoint.
- Does not train a model.
- Does not evaluate the held-out test set.
- Does not modify V1.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

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
)

from utils.experiment_data import load_experiment_data

from train_proposed import (
    probabilities,
    fuse,
)


# ============================================================
# 1. Helper functions
# ============================================================

def normalized_entropy(prob):
    """Predictive entropy normalised to the range [0, 1]."""

    p = np.clip(prob, 1e-12, 1.0)

    return -np.sum(
        p * np.log(p), axis=1
    ) / np.log(p.shape[1])


def safe_mean(values):
    """Return None rather than NaN for empty subsets."""

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return None

    return float(np.mean(values))


# ============================================================
# 2. Main diagnostic
# ============================================================

def main():

    # --------------------------------------------------------
    # Load frozen V2 data
    # --------------------------------------------------------

    train, validation, test, manifest = (
        load_experiment_data(
            str(V2 / "data/clean_budgetwise_v2.csv"),
            str(V2 / "data/experiment_split_v2.json"),
        )
    )

    assert manifest["group_key"] == "transaction_id"
    assert len(train) == 9108
    assert len(validation) == 2102
    assert len(test) == 2801

    # --------------------------------------------------------
    # Load the existing Proposed checkpoint
    # --------------------------------------------------------

    checkpoint = (
        V2
        / "outputs/validation/full/"
        "proposed_no_signal_adjustment_seed_42"
    )

    builder = joblib.load(
        checkpoint / "feature_pipeline_v2.joblib"
    )

    assert isinstance(builder, V2FeatureBuilder)

    model = MLP(108, 13)

    model.load_state_dict(
        torch.load(
            checkpoint / "best_model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Prepare validation features
    # --------------------------------------------------------

    metadata, notes, anchors, actual = (
        builder.transform_parts(validation)
    )

    n = len(validation)

    labels = builder.category_encoder.classes_

    # --------------------------------------------------------
    # Predictions before Smart Notes
    # --------------------------------------------------------

    before = probabilities(
        model,
        builder.metadata_only(metadata),
    )

    before_pred = before.argmax(axis=1)

    before_correct = before_pred == actual

    before_confidence = before.max(axis=1)

    before_entropy = normalized_entropy(before)

    # --------------------------------------------------------
    # ACTM selection
    # --------------------------------------------------------

    detector = CrossAccountConflictDetector().fit(
        train
    )

    actm = ACTM(
        ACTMConfig(
            0.65,
            0.15,
            0.30,
        )
    )

    decisions = actm.decide(
        before,
        detector.transform(validation),
    )

    selected = decisions["triggered"].to_numpy()

    note_text = (
        validation["notes"]
        .fillna("")
        .astype(str)
    )

    available = note_text.str.strip().ne("").to_numpy()

    # Notes actually accessible under ACTM.
    used = selected & available

    # TF-IDF vocabulary overlap.
    note_has_features = (
        np.asarray(
            notes.getnnz(axis=1)
        ).reshape(-1) > 0
    )

    # --------------------------------------------------------
    # Predictions after selective Smart Notes
    # --------------------------------------------------------

    fused = fuse(
        builder,
        metadata,
        notes,
        anchors,
        used,
        "semantic_anchor",
    )

    # SAME model parameters before and after.
    # Only the Smart Note feature block changes.
    after = probabilities(
        model,
        fused,
    )

    after_pred = after.argmax(axis=1)

    after_correct = after_pred == actual

    after_confidence = after.max(axis=1)

    after_entropy = normalized_entropy(after)

    # Positive value = lower predictive entropy after note.
    entropy_reduction = (
        before_entropy - after_entropy
    )

    # Positive value = increased maximum confidence.
    confidence_change = (
        after_confidence - before_confidence
    )

    # Change in probability assigned to the true category.
    row_indices = np.arange(n)

    true_label_probability_change = (
        after[row_indices, actual]
        - before[row_indices, actual]
    )

    # ========================================================
    # 3. Classify prediction changes
    # ========================================================

    wrong_to_correct = (
        ~before_correct & after_correct
    )

    correct_to_wrong = (
        before_correct & ~after_correct
    )

    wrong_stays_wrong = (
        ~before_correct & ~after_correct
    )

    correct_stays_correct = (
        before_correct & after_correct
    )

    outcomes = np.select(
        [
            wrong_to_correct,
            correct_to_wrong,
            wrong_stays_wrong,
        ],
        [
            "wrong_to_correct",
            "correct_to_wrong",
            "wrong_stays_wrong",
        ],
        default="correct_stays_correct",
    )

    # The model becomes more confident while remaining wrong.
    more_confident_still_wrong = (
        used
        & wrong_stays_wrong
        & (confidence_change > 1e-9)
    )

    # Predictive uncertainty decreases even though
    # the resulting prediction is still incorrect.
    lower_entropy_still_wrong = (
        used
        & wrong_stays_wrong
        & (entropy_reduction > 1e-9)
    )

    # Correct predictions become wrong while entropy drops.
    confident_regression = (
        used
        & correct_to_wrong
        & (entropy_reduction > 1e-9)
    )

    # ========================================================
    # 4. Same-model heuristic diagnostic
    # ========================================================

    # The training-time heuristic compares different model
    # states. Here we instead compare the SAME frozen model
    # with and without notes.
    #
    # Therefore this diagnostic score must not be described
    # as the original training-time client aggregation score.

    diagnostic_score = np.full(n, np.nan)

    novelty = np.full(n, np.nan)

    bounded_length = np.full(n, np.nan)

    heuristic_uncertainty = np.full(n, np.nan)

    if used.any():

        (
            scores,
            uncertainty_component,
            novelty_component,
            length_component,
            nonempty,
        ) = clarification_heuristic(
            before[used],
            after[used],
            notes[used],
            anchors[used],
            note_text[used].tolist(),
            weights=(0.5, 0.3, 0.2),
        )

        diagnostic_score[used] = scores

        novelty[used] = novelty_component

        bounded_length[used] = length_component

        heuristic_uncertainty[used] = (
            uncertainty_component
        )

    # ========================================================
    # 5. Per-transaction diagnostic table
    # ========================================================

    before_label = labels[before_pred]
    after_label = labels[after_pred]
    actual_label = labels[actual]

    diagnostics = pd.DataFrame({
        "source_index": validation[
            "source_index"
        ].to_numpy(),

        "transaction_id": validation[
            "transaction_id"
        ].to_numpy(),

        "actual_category": actual_label,
        "before_category": before_label,
        "after_category": after_label,

        "actm_selected": selected,
        "note_available": available,
        "note_used": used,
        "note_has_tfidf": note_has_features,

        "note_text": note_text.to_numpy(),

        "before_correct": before_correct,
        "after_correct": after_correct,
        "prediction_outcome": outcomes,

        "before_confidence": before_confidence,
        "after_confidence": after_confidence,
        "confidence_change": confidence_change,

        "before_entropy": before_entropy,
        "after_entropy": after_entropy,
        "entropy_reduction": entropy_reduction,

        "true_label_probability_change": (
            true_label_probability_change
        ),

        "more_confident_still_wrong": (
            more_confident_still_wrong
        ),

        "lower_entropy_still_wrong": (
            lower_entropy_still_wrong
        ),

        "confident_regression": confident_regression,

        "same_model_diagnostic_score": (
            diagnostic_score
        ),

        "lexical_novelty": novelty,
        "bounded_note_length": bounded_length,

        "heuristic_uncertainty_component": (
            heuristic_uncertainty
        ),
    })

    # ========================================================
    # 6. Per-category analysis
    # ========================================================

    (
        _,
        before_recall,
        before_f1,
        support,
    ) = precision_recall_fscore_support(
        actual,
        before_pred,
        labels=np.arange(13),
        zero_division=0,
    )

    (
        _,
        after_recall,
        after_f1,
        _,
    ) = precision_recall_fscore_support(
        actual,
        after_pred,
        labels=np.arange(13),
        zero_division=0,
    )

    category_rows = []

    for index, category in enumerate(labels):

        category_mask = actual == index

        prompted = category_mask & selected
        category_used = category_mask & used

        category_rows.append({
            "category": category,
            "validation_support": int(support[index]),

            "actm_prompts": int(prompted.sum()),

            "initial_errors_prompted": int(
                (prompted & ~before_correct).sum()
            ),

            "notes_used": int(category_used.sum()),

            "effective_tfidf_notes": int(
                (category_used & note_has_features).sum()
            ),

            "wrong_to_correct": int(
                (category_used & wrong_to_correct).sum()
            ),

            "correct_to_wrong": int(
                (category_used & correct_to_wrong).sum()
            ),

            "still_wrong_and_more_confident": int(
                (
                    category_used
                    & more_confident_still_wrong
                ).sum()
            ),

            "recall_before": float(before_recall[index]),
            "recall_after": float(after_recall[index]),

            "f1_before": float(before_f1[index]),
            "f1_after": float(after_f1[index]),

            "f1_change": float(
                after_f1[index] - before_f1[index]
            ),
        })

    category_table = pd.DataFrame(category_rows)

    # ========================================================
    # 7. Training-note catalogue
    # ========================================================

    # This examines source-note repetition and whether
    # the same text appears with different category labels.
    # Category diversity is a proxy for ambiguity, not
    # a definitive judgement of semantic note quality.

    training_notes = train[
        ["notes", "category"]
    ].copy()

    training_notes["note_normalized"] = (
        training_notes["notes"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    training_notes = training_notes[
        training_notes["note_normalized"].ne("")
    ]

    catalogue_rows = []

    for note_value, group in training_notes.groupby(
        "note_normalized"
    ):

        category_counts = (
            group["category"].value_counts()
        )

        catalogue_rows.append({
            "note_text": note_value,

            "frequency": len(group),

            "distinct_categories": int(
                category_counts.size
            ),

            "dominant_category": str(
                category_counts.index[0]
            ),

            "dominant_category_fraction": float(
                category_counts.iloc[0] / len(group)
            ),

            "category_distribution": json.dumps(
                category_counts.to_dict()
            ),
        })

    catalogue = pd.DataFrame(catalogue_rows)

    if not catalogue.empty:
        catalogue = catalogue.sort_values(
            "frequency",
            ascending=False,
        )

    # ========================================================
    # 8. Overall summary
    # ========================================================

    corrected_used = used & wrong_to_correct
    regressed_used = used & correct_to_wrong

    summary = {
        "experiment": "V2 Smart Note quality diagnostic",

        "checkpoint": (
            "proposed_no_signal_adjustment_seed_42"
        ),

        "evaluation_partition": "validation",

        "comparison": (
            "same frozen model with and without notes"
        ),

        "validation_rows": n,

        "test_rows_scored": 0,

        "actm_prompts": int(selected.sum()),

        "baseline_errors_targeted": int(
            (selected & ~before_correct).sum()
        ),

        "prompt_precision": float(
            (~before_correct)[selected].mean()
        ),

        "available_notes": int(available.sum()),

        "selected_notes": int(used.sum()),

        "selected_notes_with_tfidf": int(
            (used & note_has_features).sum()
        ),

        "wrong_to_correct": int(
            corrected_used.sum()
        ),

        "correct_to_wrong": int(
            regressed_used.sum()
        ),

        "net_corrections": int(
            corrected_used.sum() - regressed_used.sum()
        ),

        "still_wrong_more_confident": int(
            more_confident_still_wrong.sum()
        ),

        "still_wrong_lower_entropy": int(
            lower_entropy_still_wrong.sum()
        ),

        "regressed_with_lower_entropy": int(
            confident_regression.sum()
        ),

        "mean_entropy_reduction_on_used_notes": (
            safe_mean(entropy_reduction[used])
        ),

        "mean_confidence_change_on_used_notes": (
            safe_mean(confidence_change[used])
        ),

        "mean_true_label_probability_change": (
            safe_mean(
                true_label_probability_change[used]
            )
        ),

        "mean_heuristic_score_on_corrections": (
            safe_mean(diagnostic_score[corrected_used])
        ),

        "mean_heuristic_score_on_regressions": (
            safe_mean(diagnostic_score[regressed_used])
        ),

        "accuracy_before": float(
            accuracy_score(actual, before_pred)
        ),

        "accuracy_after": float(
            accuracy_score(actual, after_pred)
        ),

        "macro_f1_before": float(
            f1_score(
                actual,
                before_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_f1_after": float(
            f1_score(
                actual,
                after_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "training_distinct_nonempty_notes": int(
            len(catalogue)
        ),

        "training_notes_used_across_multiple_categories": int(
            (catalogue["distinct_categories"] > 1).sum()
        ) if not catalogue.empty else 0,

        "limitations": [
            "Retrospective source notes are not user-entered clarifications.",
            "Nonzero TF-IDF features do not prove semantic usefulness.",
            "Same-model diagnostic heuristic is not the original training-time aggregation heuristic.",
            "One checkpoint and one validation partition cannot establish causal or statistically reliable improvement.",
        ],
    }

    # ========================================================
    # 9. Save outputs
    # ========================================================

    output = (
        V2
        / "outputs/diagnostics/"
        "note_quality_seed_42"
    )

    output.mkdir(parents=True, exist_ok=True)

    diagnostics.loc[selected].to_csv(
        output / "selected_note_diagnostics.csv",
        index=False,
    )

    category_table.to_csv(
        output / "category_outcomes.csv",
        index=False,
    )

    catalogue.to_csv(
        output / "training_note_catalogue.csv",
        index=False,
    )

    # Export only changed predictions for manual review.
    changed = used & (
        wrong_to_correct | correct_to_wrong
    )

    diagnostics.loc[changed].to_csv(
        output / "changed_predictions.csv",
        index=False,
    )

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== V2 SMART NOTE DIAGNOSTIC ===")

    for key, value in summary.items():
        if key != "limitations":
            print(f"{key}: {value}")

    print("\n=== CATEGORY OUTCOMES ===")

    print(
        category_table[
            [
                "category",
                "actm_prompts",
                "notes_used",
                "wrong_to_correct",
                "correct_to_wrong",
                "f1_change",
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:", output)
    print("Held-out test evaluated: NO")


if __name__ == "__main__":
    main()
