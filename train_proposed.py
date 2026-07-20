"""Train and evaluate the complete proposed FYP method.

Pipeline: leakage-safe split -> metadata prediction -> ACTM -> selective Smart
Notes -> note utility -> bounded utility-weighted FedAvg -> held-out evaluation.
Run ``python train_proposed.py --help`` for reproducible experiment controls.
"""

import argparse
import copy
import json
import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset

from models.federated.actm import ACTM, ACTMConfig, CrossAccountConflictDetector
from models.federated.utility import bounded_fedavg, note_utility
from models.mlp import MLP
from utils.experiment_data import load_experiment_data, save_split_indices
from utils.metrics import metric_summary
from utils.proposed_features import ProposedFeatureBuilder


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/clean_budgetwise.csv")
    parser.add_argument("--split-manifest", default="data/experiment_split.json")
    parser.add_argument("--output", default="outputs/proposed")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--entropy-threshold", type=float, default=0.65)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--prompt-budget", type=float, default=0.30)
    parser.add_argument("--min-notes", type=int, default=1)
    parser.add_argument("--note-strategy", choices=["selective", "always", "none"], default="selective")
    parser.add_argument("--fusion-mode", choices=["semantic_anchor", "simple_concat"], default="semantic_anchor")
    parser.add_argument("--utility-weights", nargs=3, type=float, default=[0.5, 0.3, 0.2])
    parser.add_argument("--disable-utility-weighting", action="store_true")
    parser.add_argument("--max-clients", type=int, default=None,
                        help="Development/smoke-test limit; omit for the experiment.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def probabilities(model, sparse_features):
    model.eval()
    tensor = torch.as_tensor(sparse_features.toarray(), dtype=torch.float32)
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1).cpu().numpy()


def local_train(model, features, labels, epochs, learning_rate):
    tensor_x = torch.as_tensor(features.toarray(), dtype=torch.float32)
    tensor_y = torch.as_tensor(labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_sum = 0.0
    batches = 0
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item(); batches += 1
    return loss_sum / max(batches, 1)


def metric_row(actual, probs):
    predicted = probs.argmax(axis=1)
    return metric_summary(actual, predicted, probs, np.arange(probs.shape[1]))


def note_mask(frame, decisions, strategy):
    available = frame["notes"].fillna("").str.strip().ne("").to_numpy()
    if strategy == "selective":
        return decisions["triggered"].to_numpy() & available
    if strategy == "always":
        return available
    if strategy == "none":
        return np.zeros(len(frame), dtype=bool)
    raise ValueError(f"Unknown note strategy: {strategy}")


def fuse(features, metadata, notes, anchors, mask, mode):
    if mode == "semantic_anchor":
        return features.semantic_anchor_fused(metadata, notes, anchors, mask)
    if mode == "simple_concat":
        return features.fused(metadata, notes, mask)
    raise ValueError(f"Unknown fusion mode: {mode}")


def main(config):
    set_seed(config.seed)
    if any(weight < 0 for weight in config.utility_weights) or not np.isclose(sum(config.utility_weights), 1.0):
        raise ValueError("utility_weights must be non-negative and sum to 1.0")
    output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    train, validation, test, manifest = load_experiment_data(
        config.dataset, config.split_manifest
    )
    save_split_indices(output, {"train": train, "validation": validation, "test": test})

    features = ProposedFeatureBuilder().fit(train)
    conflict_detector = CrossAccountConflictDetector().fit(train)
    actm = ACTM(ACTMConfig(
        config.entropy_threshold, config.margin_threshold, config.prompt_budget
    ))
    input_size = features.metadata_size + features.note_size
    classes = len(features.category_encoder.classes_)
    global_model = MLP(input_size, classes)

    client_ids = sorted(train["user_id"].fillna("Unknown").astype(str).unique())
    if config.max_clients:
        client_ids = client_ids[:config.max_clients]
        train = train[train["user_id"].fillna("Unknown").astype(str).isin(client_ids)]
    round_rows, weight_rows, utility_rows, trigger_rows = [], [], [], []
    best_f1 = -1.0; checkpoint = output / "best_global_model.pt"

    for round_number in range(1, config.rounds + 1):
        client_results = []
        for client_id in client_ids:
            client_frame = train[train["user_id"].fillna("Unknown").astype(str) == client_id]
            metadata, notes, anchors, labels = features.transform_parts(client_frame)
            before = probabilities(global_model, features.metadata_only(metadata))
            decisions = actm.decide(before, conflict_detector.transform(client_frame))
            use_notes = note_mask(client_frame, decisions, config.note_strategy)
            fused = fuse(features, metadata, notes, anchors, use_notes, config.fusion_mode)
            local_model = copy.deepcopy(global_model)
            local_loss = local_train(local_model, fused, labels, config.local_epochs, config.learning_rate)
            after = probabilities(local_model, fused)
            selected = np.flatnonzero(use_notes)
            if len(selected):
                utility, reduction, specificity, effort, nonempty = note_utility(
                    before[selected], after[selected], notes[selected], anchors[selected],
                    client_frame.iloc[selected]["notes"].tolist(), weights=tuple(config.utility_weights),
                )
                valid_utility = utility[nonempty]
                mean_utility = float(valid_utility.mean()) if len(valid_utility) else None
                for position, row_index in enumerate(selected):
                    utility_rows.append({
                        "round": round_number, "client_id": client_id,
                        "transaction_id": client_frame.iloc[row_index]["transaction_id"],
                        "uncertainty_reduction": reduction[position],
                        "semantic_specificity": specificity[position],
                        "bounded_effort": effort[position], "utility": utility[position],
                    })
            else:
                mean_utility = None
            decisions = decisions.assign(
                round=round_number, client_id=client_id,
                transaction_id=client_frame["transaction_id"].to_numpy(), note_used=use_notes,
            )
            trigger_rows.extend(decisions.to_dict("records"))
            client_results.append({
                "client_id": client_id, "weights": copy.deepcopy(local_model.state_dict()),
                "sample_count": len(client_frame), "note_count": int(use_notes.sum()),
                "mean_note_utility": mean_utility, "local_loss": local_loss,
            })

        aggregation_results = client_results
        if config.disable_utility_weighting:
            aggregation_results = [
                {**result, "mean_note_utility": None, "note_count": 0}
                for result in client_results
            ]
        weights, diagnostics = bounded_fedavg(aggregation_results, min_notes=config.min_notes)
        global_model.load_state_dict(weights)
        for result, diagnostic in zip(client_results, diagnostics):
            weight_rows.append({"round": round_number, "client_id": result["client_id"],
                                "local_loss": result["local_loss"], **diagnostic})
        val_metadata, val_notes, val_anchors, val_labels = features.transform_parts(validation)
        val_before = probabilities(global_model, features.metadata_only(val_metadata))
        val_decisions = actm.decide(val_before, conflict_detector.transform(validation))
        val_mask = note_mask(validation, val_decisions, config.note_strategy)
        val_probs = probabilities(global_model, fuse(
            features, val_metadata, val_notes, val_anchors, val_mask, config.fusion_mode
        ))
        metrics = metric_row(val_labels, val_probs)
        round_rows.append({"round": round_number, "clients": len(client_results),
                           "prompts_per_100": 100 * val_decisions["triggered"].mean(), **metrics})
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]; torch.save(global_model.state_dict(), checkpoint)
        print(f"Round {round_number}/{config.rounds}: validation macro-F1={metrics['macro_f1']:.4f}")

    # Final outputs always represent the validation-selected checkpoint.
    global_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    metadata, notes, anchors, actual = features.transform_parts(test)
    before = probabilities(global_model, features.metadata_only(metadata))
    decisions = actm.decide(before, conflict_detector.transform(test))
    test_note_mask = note_mask(test, decisions, config.note_strategy)
    final_probs = probabilities(global_model, fuse(
        features, metadata, notes, anchors, test_note_mask, config.fusion_mode
    ))
    predicted = final_probs.argmax(axis=1); overall = metric_row(actual, final_probs)
    ambiguous = decisions["eligible"].to_numpy()
    ambiguous_metrics = metric_row(actual[ambiguous], final_probs[ambiguous]) if ambiguous.any() else {}

    pd.DataFrame(round_rows).to_csv(output / "round_metrics.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output / "aggregation_weights.csv", index=False)
    pd.DataFrame(utility_rows).to_csv(output / "utility_scores.csv", index=False)
    pd.DataFrame(trigger_rows).to_csv(output / "actm_triggers.csv", index=False)
    confidence = final_probs.max(axis=1); correct = predicted == actual
    calibration_rows = []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        calibration_rows.append({
            "bin_lower": lower, "bin_upper": upper, "count": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
            "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
        })
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_metrics.csv", index=False)
    pd.DataFrame([{"model": "Proposed", **overall}]).to_csv(output / "overall_metrics.csv", index=False)
    pd.DataFrame([{"model": "Proposed", "ambiguous_count": int(ambiguous.sum()), **ambiguous_metrics}]).to_csv(output / "ambiguous_metrics.csv", index=False)
    pd.DataFrame([{
        "strategy": "ACTM", "transactions": len(test), "eligible": int(decisions["eligible"].sum()),
        "prompts": int(decisions["triggered"].sum()), "notes_used": int(test_note_mask.sum()),
        "prompts_per_100": 100 * decisions["triggered"].mean(),
        "prompt_precision": float((before.argmax(axis=1)[decisions["triggered"].to_numpy()] != actual[decisions["triggered"].to_numpy()]).mean()) if decisions["triggered"].any() else 0,
        "note_acceptance_rate": float(test_note_mask.sum() / max(decisions["triggered"].sum(), 1)),
        "mean_uncertainty_reduction": float(np.maximum(
            0.0, -(before * np.log(np.clip(before, 1e-12, 1))).sum(axis=1)
            + (final_probs * np.log(np.clip(final_probs, 1e-12, 1))).sum(axis=1)
        )[test_note_mask].mean()) if test_note_mask.any() else 0.0,
    }]).to_csv(output / "prompt_metrics.csv", index=False)
    pd.DataFrame({
        "transaction_id": test["transaction_id"].to_numpy(), "actual": actual, "predicted": predicted,
        "actual_label": features.category_encoder.inverse_transform(actual),
        "predicted_label": features.category_encoder.inverse_transform(predicted),
        "confidence": final_probs.max(axis=1), "actm_triggered": decisions["triggered"].to_numpy(),
        "note_used": test_note_mask,
    }).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(final_probs).to_csv(output / "class_probabilities.csv", index=False)
    (output / "classification_report.txt").write_text(
        classification_report(actual, predicted, target_names=features.category_encoder.classes_, zero_division=0),
        encoding="utf-8",
    )
    pd.DataFrame({"category": features.category_encoder.classes_, "encoded": range(classes)}).to_csv(output / "category_mapping.csv", index=False)
    joblib.dump(features, output / "feature_pipeline.pkl")
    model_config = {"input_features": input_size, "output_classes": classes, "hidden_layers": [128, 64]}
    (output / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    experiment = {
        "method": "ACTM + selective Smart Notes + note utility + bounded utility-weighted FedAvg",
        "split": {"train": len(train), "validation": len(validation), "test": len(test),
                  "seed": manifest["seed"], "manifest_version": manifest["version"]},
        "actm": vars(actm.config), "note_strategy": config.note_strategy,
        "fusion_mode": config.fusion_mode, "utility_weights": list(config.utility_weights),
        "utility_weighting_enabled": not config.disable_utility_weighting,
        "utility_multiplier_bounds": [0.75, 1.25], "minimum_notes_for_weighting": config.min_notes,
        "cross_account_proxy": {"merchant": "location", "account": "payment_mode"},
        "rounds": config.rounds, "local_epochs": config.local_epochs,
        "training_seed": config.seed,
        "best_validation_macro_f1": best_f1, "final_test_metrics": overall,
    }
    (output / "experiment_info.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    cm = confusion_matrix(actual, predicted, labels=np.arange(classes))
    plt.figure(figsize=(9, 7)); plt.imshow(cm, cmap="viridis"); plt.colorbar()
    plt.title("Proposed Method Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(output / "confusion_matrix.png", dpi=160); plt.close()
    rounds = pd.DataFrame(round_rows)
    plt.figure(figsize=(8, 5)); plt.plot(rounds["round"], rounds["macro_f1"], marker="o")
    plt.xlabel("Communication round"); plt.ylabel("Validation macro-F1"); plt.tight_layout()
    plt.savefig(output / "convergence_curve.png", dpi=160); plt.close()
    calibration = pd.DataFrame(calibration_rows).dropna()
    plt.figure(figsize=(6, 6)); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(calibration["mean_confidence"], calibration["accuracy"], marker="o")
    plt.xlabel("Mean confidence"); plt.ylabel("Observed accuracy"); plt.tight_layout()
    plt.savefig(output / "reliability_diagram.png", dpi=160); plt.close()
    if utility_rows:
        plt.figure(figsize=(7, 5)); plt.hist(pd.DataFrame(utility_rows)["utility"], bins=20)
        plt.xlabel("Smart Note utility"); plt.ylabel("Count"); plt.tight_layout()
        plt.savefig(output / "utility_distribution.png", dpi=160); plt.close()
    print(f"Final held-out test macro-F1={overall['macro_f1']:.4f}; outputs: {output}")


if __name__ == "__main__":
    main(arguments())
