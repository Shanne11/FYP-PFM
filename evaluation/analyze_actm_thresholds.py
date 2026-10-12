"""Diagnose ACTM eligibility thresholds on the validation partition only.

This script does not tune on or report held-out test outcomes.  It explains why
the frozen ACTM eligibility conditions cover every record and shows how stricter
entropy/margin pairs behave with and without the current cross-context proxy.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.federated.actm import (  # noqa: E402
    CrossAccountConflictDetector,
    predictive_entropy,
    top_two_margin,
)
from models.mlp import MLP  # noqa: E402
from train_proposed import probabilities  # noqa: E402
from utils.experiment_data import load_experiment_data  # noqa: E402


DEFAULT_SEEDS = (42, 52, 62)
ENTROPY_THRESHOLDS = (0.65, 0.95, 0.975, 0.985, 0.99, 0.995, 0.9975, 0.999)
MARGIN_THRESHOLDS = (0.15, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/clean_budgetwise.csv")
    parser.add_argument("--split-manifest", default="data/experiment_split.json")
    parser.add_argument("--checkpoint-root", default="outputs/ablations")
    parser.add_argument("--output", default="outputs/actm_threshold_analysis")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--prompt-budget", type=float, default=0.30)
    return parser.parse_args()


def load_checkpoint(path):
    features = joblib.load(path / "feature_pipeline.pkl")
    config = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
    model = MLP(config["input_features"], config["output_classes"])
    model.load_state_dict(
        torch.load(path / "best_global_model.pt", map_location="cpu", weights_only=True)
    )
    return features, model


def eligibility_metrics(
    normalized_entropy,
    margin,
    conflicts,
    entropy_threshold,
    margin_threshold,
    include_conflict,
    prompt_budget,
):
    entropy_hit = normalized_entropy >= entropy_threshold
    margin_hit = margin <= margin_threshold
    conflict_hit = conflicts if include_conflict else np.zeros_like(conflicts, dtype=bool)
    eligible = entropy_hit | margin_hit | conflict_hit
    budget_count = int(np.ceil(len(eligible) * prompt_budget))
    return {
        "entropy_threshold": entropy_threshold,
        "margin_threshold": margin_threshold,
        "include_conflict": include_conflict,
        "entropy_hit_rate": float(entropy_hit.mean()),
        "margin_hit_rate": float(margin_hit.mean()),
        "conflict_hit_rate": float(conflict_hit.mean()),
        "eligibility_count": int(eligible.sum()),
        "eligibility_rate": float(eligible.mean()),
        "budget_count": budget_count,
        "budget_fill_rate": min(int(eligible.sum()), budget_count) / budget_count,
        "expected_prompt_rate": min(int(eligible.sum()), budget_count) / len(eligible),
    }


def main(config):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    train, validation, _, manifest = load_experiment_data(
        config.dataset, config.split_manifest
    )
    detector = CrossAccountConflictDetector().fit(train)
    validation_conflicts = detector.transform(validation)
    distribution_rows = []
    sensitivity_rows = []

    for seed in config.seeds:
        checkpoint = Path(config.checkpoint_root) / f"seed_{seed}" / "full"
        features, model = load_checkpoint(checkpoint)
        metadata, _, _, _ = features.transform_parts(validation)
        predicted = probabilities(model, features.metadata_only(metadata))
        normalized_entropy = predictive_entropy(predicted) / np.log(predicted.shape[1])
        margin = top_two_margin(predicted)
        distribution_rows.append(
            {
                "seed": seed,
                "validation_rows": len(validation),
                "conflict_rate": float(validation_conflicts.mean()),
                **{
                    f"entropy_q{int(q * 100):02d}": float(np.quantile(normalized_entropy, q))
                    for q in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
                },
                **{
                    f"margin_q{int(q * 100):02d}": float(np.quantile(margin, q))
                    for q in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
                },
            }
        )
        for include_conflict in (True, False):
            for entropy_threshold in ENTROPY_THRESHOLDS:
                for margin_threshold in MARGIN_THRESHOLDS:
                    sensitivity_rows.append(
                        {
                            "seed": seed,
                            **eligibility_metrics(
                                normalized_entropy,
                                margin,
                                validation_conflicts,
                                entropy_threshold,
                                margin_threshold,
                                include_conflict,
                                config.prompt_budget,
                            ),
                        }
                    )

    distributions = pd.DataFrame(distribution_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    summary = (
        sensitivity.groupby(
            ["include_conflict", "entropy_threshold", "margin_threshold"],
            sort=False,
        )
        .agg(
            eligibility_rate_mean=("eligibility_rate", "mean"),
            eligibility_rate_std=("eligibility_rate", "std"),
            expected_prompt_rate_mean=("expected_prompt_rate", "mean"),
            budget_fill_rate_mean=("budget_fill_rate", "mean"),
        )
        .reset_index()
    )
    distributions.to_csv(output / "actm_threshold_distributions.csv", index=False)
    sensitivity.to_csv(output / "actm_threshold_sensitivity_runs.csv", index=False)
    summary.to_csv(output / "actm_threshold_sensitivity_summary.csv", index=False)
    current = summary[
        (summary["include_conflict"])
        & (summary["entropy_threshold"] == 0.65)
        & (summary["margin_threshold"] == 0.15)
    ].iloc[0]
    current_without_conflict = summary[
        (~summary["include_conflict"])
        & (summary["entropy_threshold"] == 0.65)
        & (summary["margin_threshold"] == 0.15)
    ].iloc[0]
    selective_example = summary[
        (~summary["include_conflict"])
        & (summary["entropy_threshold"] == 0.999)
        & (summary["margin_threshold"] == 0.001)
    ].iloc[0]
    conclusion = (
        "ACTM validation-only threshold diagnosis\n\n"
        f"Frozen threshold eligibility with the conflict proxy: "
        f"{current['eligibility_rate_mean']:.2%}.\n"
        f"Frozen threshold eligibility without the conflict proxy: "
        f"{current_without_conflict['eligibility_rate_mean']:.2%}.\n"
        f"Training-fitted proxy conflict rate on validation: "
        f"{distributions['conflict_rate'].mean():.2%}.\n"
        f"Normalised entropy range across seeds: "
        f"{distributions['entropy_q00'].min():.6f} to "
        f"{distributions['entropy_q100'].max():.6f}.\n"
        f"Top-two margin range across seeds: "
        f"{distributions['margin_q00'].min():.6f} to "
        f"{distributions['margin_q100'].max():.6f}.\n"
        "Illustrative validation-only pair without conflict "
        "(entropy >= 0.999 or margin <= 0.001): "
        f"{selective_example['eligibility_rate_mean']:.2%} mean eligibility "
        f"(+/- {selective_example['eligibility_rate_std']:.2%}).\n\n"
        "The current 0.65 entropy and 0.15 margin thresholds are non-selective "
        "for these near-uniform model outputs. The location/payment-mode conflict "
        "proxy is also degenerate because it marks every validation record. "
        "Threshold changes alone cannot restore selectivity while that proxy is "
        "included. Any replacement thresholds or conflict definition must be "
        "chosen on validation data, frozen, and then evaluated once on held-out "
        "data; this diagnostic does not alter the reported experimental contract.\n"
    )
    (output / "actm_threshold_conclusion.txt").write_text(
        conclusion, encoding="utf-8"
    )
    print(conclusion)


if __name__ == "__main__":
    main(arguments())
