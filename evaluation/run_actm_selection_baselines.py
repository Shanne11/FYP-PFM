"""Evaluate budget-matched ACTM selection policies using frozen checkpoints.

This is a selection-policy comparison, not a retraining experiment.  For each
seed, every strategy uses the same validation-selected Proposed checkpoint,
held-out records, nominal prompt budget, available source-note proxies and
same-model before/after inference path.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.federated.actm import (  # noqa: E402
    ACTM,
    ACTMConfig,
    CrossAccountConflictDetector,
    budget_matched_selection_baselines,
)
from models.mlp import MLP  # noqa: E402
from train_proposed import probabilities  # noqa: E402
from utils.experiment_data import load_experiment_data  # noqa: E402


DEFAULT_SEEDS = (42, 52, 62)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/clean_budgetwise.csv")
    parser.add_argument("--split-manifest", default="data/experiment_split.json")
    parser.add_argument("--checkpoint-root", default="outputs/ablations")
    parser.add_argument("--output", default="outputs/actm_selection_baselines")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--prompt-budget", type=float, default=0.30)
    parser.add_argument("--entropy-threshold", type=float, default=0.65)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    return parser.parse_args()


def load_checkpoint(path):
    features = joblib.load(path / "feature_pipeline.pkl")
    config = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
    model = MLP(config["input_features"], config["output_classes"])
    model.load_state_dict(
        torch.load(path / "best_global_model.pt", map_location="cpu", weights_only=True)
    )
    return features, model


def strategy_row(seed, strategy, prompted, note_used, actual, before, after):
    before_predicted, after_predicted = before.argmax(1), after.argmax(1)
    before_correct, after_correct = before_predicted == actual, after_predicted == actual
    before_confidence, after_confidence = before.max(1), after.max(1)
    evaluated = np.asarray(note_used, dtype=bool)
    true_change = after[np.arange(len(actual)), actual] - before[
        np.arange(len(actual)), actual
    ]
    return {
        "seed": seed,
        "strategy": strategy,
        "transactions": len(actual),
        "prompts": int(prompted.sum()),
        "prompts_per_100": 100.0 * float(prompted.mean()),
        "notes_used": int(evaluated.sum()),
        "note_acceptance_rate": float(evaluated.sum() / max(prompted.sum(), 1)),
        "selected_initially_wrong": int((prompted & ~before_correct).sum()),
        "prompt_precision": float((~before_correct[prompted]).mean()),
        "wrong_to_correct": int((evaluated & ~before_correct & after_correct).sum()),
        "correct_to_wrong": int((evaluated & before_correct & ~after_correct).sum()),
        "wrong_to_different_wrong": int(
            (evaluated & ~before_correct & ~after_correct & (before_predicted != after_predicted)).sum()
        ),
        "wrong_unchanged": int(
            (evaluated & ~before_correct & ~after_correct & (before_predicted == after_predicted)).sum()
        ),
        "correct_unchanged": int((evaluated & before_correct & after_correct).sum()),
        "more_confident_still_wrong": int(
            (evaluated & ~before_correct & ~after_correct & (after_confidence > before_confidence)).sum()
        ),
        "net_corrections": int((evaluated & ~before_correct & after_correct).sum())
        - int((evaluated & before_correct & ~after_correct).sum()),
        "correction_yield_per_100_prompts": 100.0
        * float((evaluated & ~before_correct & after_correct).sum())
        / max(prompted.sum(), 1),
        "mean_true_class_probability_change": float(true_change[evaluated].mean())
        if evaluated.any()
        else 0.0,
        "initial_note_subset_accuracy": float(before_correct[evaluated].mean())
        if evaluated.any()
        else 0.0,
        "final_note_subset_accuracy": float(after_correct[evaluated].mean())
        if evaluated.any()
        else 0.0,
        "overall_accuracy": accuracy_score(actual, after_predicted),
        "overall_macro_f1": f1_score(
            actual, after_predicted, average="macro", zero_division=0
        ),
    }


def outcome_rows(seed, strategy, frame, actual, prompted, note_used, before, after):
    before_predicted, after_predicted = before.argmax(1), after.argmax(1)
    return pd.DataFrame(
        {
            "seed": seed,
            "strategy": strategy,
            "source_index": frame["source_index"].to_numpy(),
            "transaction_id": frame["transaction_id"].to_numpy(),
            "prompted": prompted,
            "note_available": frame["notes"].fillna("").str.strip().ne("").to_numpy(),
            "note_used": note_used,
            "actual": actual,
            "predicted_before": before_predicted,
            "predicted_after": after_predicted,
            "correct_before": before_predicted == actual,
            "correct_after": after_predicted == actual,
            "confidence_before": before.max(1),
            "confidence_after": after.max(1),
            "true_probability_before": before[np.arange(len(actual)), actual],
            "true_probability_after": after[np.arange(len(actual)), actual],
        }
    )


def summarize(runs):
    numeric = [
        column
        for column in runs.select_dtypes(include="number").columns
        if column != "seed"
    ]
    rows = []
    for strategy, group in runs.groupby("strategy", sort=False):
        row = {"strategy": strategy, "runs": len(group)}
        for column in numeric:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_std"] = group[column].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main(config):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    train, _, test, manifest = load_experiment_data(config.dataset, config.split_manifest)
    record_ids = test["source_index"].to_numpy()
    available_notes = test["notes"].fillna("").str.strip().ne("").to_numpy()
    run_rows, transaction_frames, overlap_rows = [], [], []

    for seed in config.seeds:
        checkpoint = Path(config.checkpoint_root) / f"seed_{seed}" / "full"
        features, model = load_checkpoint(checkpoint)
        metadata, notes, anchors, actual = features.transform_parts(test)
        before = probabilities(model, features.metadata_only(metadata))
        conflict_detector = CrossAccountConflictDetector().fit(train)
        conflicts = conflict_detector.transform(test)
        actm = ACTM(
            ACTMConfig(
                config.entropy_threshold,
                config.margin_threshold,
                config.prompt_budget,
            )
        )
        proposed = actm.decide(before, conflicts)["triggered"].to_numpy()
        masks = budget_matched_selection_baselines(
            before, conflicts, record_ids, config.prompt_budget, seed
        )
        masks = {**masks, "Proposed ACTM": proposed}

        for left, right in combinations(masks, 2):
            intersection = int((masks[left] & masks[right]).sum())
            union = int((masks[left] | masks[right]).sum())
            overlap_rows.append(
                {
                    "seed": seed,
                    "strategy_a": left,
                    "strategy_b": right,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 1.0,
                }
            )

        for strategy, prompted in masks.items():
            note_used = prompted & available_notes
            fused = features.semantic_anchor_fused(
                metadata, notes, anchors, note_used
            )
            after = probabilities(model, fused)
            run_rows.append(
                strategy_row(
                    seed, strategy, prompted, note_used, actual, before, after
                )
            )
            transaction_frames.append(
                outcome_rows(
                    seed,
                    strategy,
                    test,
                    actual,
                    prompted,
                    note_used,
                    before,
                    after,
                )
            )

    runs = pd.DataFrame(run_rows)
    summary = summarize(runs)
    runs.to_csv(output / "actm_selection_runs.csv", index=False)
    summary.to_csv(output / "actm_selection_summary.csv", index=False)
    pd.concat(transaction_frames, ignore_index=True).to_csv(
        output / "actm_selection_transaction_outcomes.csv", index=False
    )
    pd.DataFrame(overlap_rows).to_csv(
        output / "actm_selection_overlap.csv", index=False
    )
    (output / "actm_selection_contract.json").write_text(
        json.dumps(
            {
                "evaluation_role": "same-checkpoint held-out selection-policy comparison",
                "split_manifest": str(config.split_manifest),
                "split_seed": manifest["seed"],
                "training_seeds": list(config.seeds),
                "prompt_budget": config.prompt_budget,
                "budget_rounding": "ceil",
                "strategies": list(runs["strategy"].drop_duplicates()),
                "checkpoint_pattern": str(
                    Path(config.checkpoint_root) / "seed_<seed>" / "full"
                ),
                "conflict_proxy": {
                    "merchant": "location",
                    "account_or_channel": "payment_mode",
                },
                "raw_notes_transmitted": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    proposed = summary[summary["strategy"] == "Proposed ACTM"].iloc[0]
    best_gross = summary.sort_values(
        ["correction_yield_per_100_prompts_mean", "correct_to_wrong_mean"],
        ascending=[False, True],
    ).iloc[0]
    best_net = summary.sort_values(
        ["net_corrections_mean", "overall_macro_f1_mean"],
        ascending=[False, False],
    ).iloc[0]
    best_precision = summary.sort_values(
        "prompt_precision_mean", ascending=False
    ).iloc[0]
    (output / "actm_selection_conclusion.txt").write_text(
        "ACTM selection comparison completed under a shared nominal 30% prompt "
        "budget and frozen same-checkpoint inference. Results must be interpreted "
        "as retrospective source-note-proxy evidence, not observed human response.\n"
        f"Highest gross correction yield: {best_gross['strategy']} "
        f"({best_gross['correction_yield_per_100_prompts_mean']:.3f} wrong-to-correct "
        "changes per 100 prompts).\n"
        f"Highest mean net corrections: {best_net['strategy']} "
        f"({best_net['net_corrections_mean']:.3f} per seed).\n"
        f"Highest prompt precision: {best_precision['strategy']} "
        f"({best_precision['prompt_precision_mean']:.6f}).\n"
        f"Proposed ACTM: prompt precision={proposed['prompt_precision_mean']:.6f}, "
        f"gross correction yield={proposed['correction_yield_per_100_prompts_mean']:.3f}, "
        f"net corrections={proposed['net_corrections_mean']:.3f} per seed.\n"
        "No strategy dominates every outcome; report the trade-off rather than an "
        "unqualified ACTM-superiority claim.\n",
        encoding="utf-8",
    )
    print(runs.to_string(index=False))
    print("\nThree-seed summary\n", summary.to_string(index=False))


if __name__ == "__main__":
    main(arguments())
