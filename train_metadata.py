"""Baseline 2: leakage-safe metadata-only Random Forest."""

from pathlib import Path

import joblib
import pandas as pd

from models.metadata_model import build_model
from utils.experiment_data import load_experiment_data, save_split_indices
from utils.metrics import evaluate
from utils.proposed_features import ProposedFeatureBuilder


OUTPUT = Path("outputs/baseline2"); OUTPUT.mkdir(parents=True, exist_ok=True)
train, validation, test, manifest = load_experiment_data()
save_split_indices(OUTPUT, {"train": train, "validation": validation, "test": test})
builder = ProposedFeatureBuilder().fit(train)
X_train, _, _, y_train = builder.transform_parts(train)
X_test, _, _, y_test = builder.transform_parts(test)
model = build_model(); model.fit(X_train, y_train); predicted = model.predict(X_test)
actual_labels = builder.category_encoder.inverse_transform(y_test)
predicted_labels = builder.category_encoder.inverse_transform(predicted)
metrics = evaluate(actual_labels, predicted_labels, str(OUTPUT))

feature_names = list(builder.categorical_encoder.get_feature_names_out()) + builder.numeric
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
