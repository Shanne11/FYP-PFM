import os
import pandas as pd

from models.rules import predict
from utils.metrics import evaluate

OUTPUT_FOLDER = "outputs/baseline1"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("dataset/clean_budgetwise.csv")

actual = []
predicted = []

for _, row in df.iterrows():

    actual.append(row["category"])

    predicted.append(
        predict(row["notes"])
    )

# Save predictions
df["predicted_category"] = predicted

df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "predictions.csv"
    ),
    index=False
)

# Evaluate and save metrics
evaluate(
    actual,
    predicted,
    OUTPUT_FOLDER
)