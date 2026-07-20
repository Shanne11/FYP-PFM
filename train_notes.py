"""Baseline 3: metadata plus all available Smart Notes."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

from models.note_model import build_model
from utils.experiment_data import load_experiment_data, save_split_indices
from utils.metrics import evaluate
from utils.proposed_features import ProposedFeatureBuilder


OUTPUT = Path("outputs/baseline3"); OUTPUT.mkdir(parents=True, exist_ok=True)
train, validation, test, manifest = load_experiment_data()
save_split_indices(OUTPUT, {"train": train, "validation": validation, "test": test})
builder = ProposedFeatureBuilder().fit(train)
train_meta, train_notes, _, y_train = builder.transform_parts(train)
test_meta, test_notes, _, y_test = builder.transform_parts(test)
X_train = hstack([train_meta, train_notes], format="csr")
X_test = hstack([test_meta, test_notes], format="csr")
model = build_model(); model.fit(X_train, y_train); predicted = model.predict(X_test)
probabilities = model.predict_proba(X_test)
actual_labels = builder.category_encoder.inverse_transform(y_test)
predicted_labels = builder.category_encoder.inverse_transform(predicted)
metrics = evaluate(
    actual_labels, predicted_labels, str(OUTPUT), probabilities,
    builder.category_encoder.inverse_transform(model.classes_),
)
feature_names = list(builder.categorical_encoder.get_feature_names_out()) + builder.numeric + list(builder.note_vectorizer.get_feature_names_out())
pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(
    "importance", ascending=False
).to_csv(OUTPUT / "feature_importance.csv", index=False)
pd.DataFrame({
    "transaction_id": test["transaction_id"], "actual": actual_labels,
    "predicted": predicted_labels,
}).to_csv(OUTPUT / "predictions.csv", index=False)
joblib.dump(model, OUTPUT / "note_model.pkl"); joblib.dump(builder, OUTPUT / "feature_pipeline.pkl")
(OUTPUT / "experiment_info.txt").write_text(
    f"Baseline: Metadata + Notes Random Forest\nTrain: {len(train)}\nValidation: {len(validation)}\n"
    f"Test: {len(test)}\nClasses: {len(builder.category_encoder.classes_)}\n"
    f"Split manifest version: {manifest['version']}\nMetrics: {metrics}\n",
    encoding="utf-8",
)
print(metrics)
