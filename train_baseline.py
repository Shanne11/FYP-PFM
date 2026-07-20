"""Baseline 1: rules-only categorisation on the shared held-out test set."""

from pathlib import Path

import pandas as pd

from models.rules import predict
from utils.experiment_data import load_experiment_data, save_split_indices
from utils.metrics import evaluate


OUTPUT = Path("outputs/baseline1")
train, validation, test, manifest = load_experiment_data()
save_split_indices(OUTPUT, {"train": train, "validation": validation, "test": test})

predicted = test["notes"].map(predict)
pd.DataFrame({
    "transaction_id": test["transaction_id"],
    "actual": test["category"],
    "predicted": predicted,
}).to_csv(OUTPUT / "predictions.csv", index=False)
evaluate(test["category"].to_numpy(), predicted.to_numpy(), str(OUTPUT))
(OUTPUT / "experiment_info.txt").write_text(
    f"Baseline: Rules-only\nTrain: 0\nValidation: 0\nTest: {len(test)}\n"
    f"Split manifest version: {manifest['version']}\n",
    encoding="utf-8",
)
