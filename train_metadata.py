"""Baseline 2: leakage-safe metadata-only Random Forest."""

import argparse

from pathlib import Path

import joblib
import pandas as pd

from models.metadata_model import build_model
from utils.experiment_data import load_experiment_data, save_split_indices
from utils.metrics import evaluate
from utils.proposed_features import ProposedFeatureBuilder


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/clean_budgetwise.csv")
    parser.add_argument("--split-manifest", default="data/experiment_split.json")
    parser.add_argument("--output", default="outputs/baseline2")
    return parser.parse_args()


config = arguments()
OUTPUT = Path(config.output); OUTPUT.mkdir(parents=True, exist_ok=True)
train, validation, test, manifest = load_experiment_data(config.dataset, config.split_manifest)
save_split_indices(OUTPUT, {"train": train, "validation": validation, "test": test})
builder = ProposedFeatureBuilder().fit(train)
X_train, _, _, y_train = builder.transform_parts(train)
X_test, _, _, y_test = builder.transform_parts(test)
if builder.metadata_size != 68 or len(builder.category_encoder.classes_) != 13:
    raise ValueError("B1 requires the fixed 68-feature, 13-category Tier A metadata contract")
model = build_model(); model.fit(X_train, y_train); predicted = model.predict(X_test)
probabilities = model.predict_proba(X_test)
actual_labels = builder.category_encoder.inverse_transform(y_test)
predicted_labels = builder.category_encoder.inverse_transform(predicted)
metrics = evaluate(
    actual_labels, predicted_labels, str(OUTPUT), probabilities,
    builder.category_encoder.inverse_transform(model.classes_),
)

feature_names = builder.metadata_feature_names
pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(
    "importance", ascending=False
).to_csv(OUTPUT / "feature_importance.csv", index=False)
pd.DataFrame({
    "transaction_id": test["transaction_id"], "actual": actual_labels,
    "predicted": predicted_labels,
}).to_csv(OUTPUT / "predictions.csv", index=False)
pd.DataFrame({"category": builder.category_encoder.classes_, "encoded": range(len(builder.category_encoder.classes_))}).to_csv(
    OUTPUT / "category_mapping.csv", index=False
)
joblib.dump(model, OUTPUT / "metadata_model.pkl"); joblib.dump(builder, OUTPUT / "feature_pipeline.pkl")
(OUTPUT / "experiment_info.txt").write_text(
    f"Baseline: Metadata-only Random Forest\nTrain: {len(train)}\nValidation: {len(validation)}\n"
    f"Test: {len(test)}\nClasses: {len(builder.category_encoder.classes_)}\n"
    f"Split manifest version: {manifest['version']}\nMetrics: {metrics}\n",
    encoding="utf-8",
)
print(metrics)
