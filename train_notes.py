import os
import joblib
import pandas as pd

from utils.feature_engineering import prepare_metadata_note_features
from utils.metrics import evaluate
from models.note_model import build_model

OUTPUT = "outputs/baseline3"

os.makedirs(OUTPUT, exist_ok=True)

print("=" * 60)
print("Baseline 3 - Metadata + Notes")
print("=" * 60)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("dataset/clean_budgetwise.csv")

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

print("Preparing metadata + note features...")

X, y, encoders, vectorizer = prepare_metadata_note_features(df)

# --------------------------------------------------
# Load Train/Test Split from Baseline 2
# --------------------------------------------------

print("Loading train/test split from Baseline 2...")

train_index = pd.read_csv(
    "outputs/baseline2/train_indices.csv",
    header=None
)[0]

test_index = pd.read_csv(
    "outputs/baseline2/test_indices.csv",
    header=None
)[0]

X_train = X[train_index]
X_test = X[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]


# --------------------------------------------------
# Build Model
# --------------------------------------------------

print("Building model...")

model = build_model()

# --------------------------------------------------
# Train
# --------------------------------------------------

print("Training model...")

model.fit(X_train, y_train)

# --------------------------------------------------
# Predict
# --------------------------------------------------

print("Predicting...")

prediction = model.predict(X_test)

# --------------------------------------------------
# Save Model
# --------------------------------------------------

joblib.dump(
    model,
    os.path.join(
        OUTPUT,
        "note_model.pkl"
    )
)

joblib.dump(
    vectorizer,
    os.path.join(
        OUTPUT,
        "tfidf_vectorizer.pkl"
    )
)

joblib.dump(
    encoders,
    os.path.join(
        OUTPUT,
        "label_encoders.pkl"
    )
)

# --------------------------------------------------
# Save Predictions
# --------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": prediction
})

results.to_csv(
    os.path.join(
        OUTPUT,
        "predictions.csv"
    ),
    index=False
)

# --------------------------------------------------
# Save Feature Importance
# --------------------------------------------------

try:

    feature_names = [
        "transaction_type",
        "amount",
        "payment_mode",
        "location",
        "year",
        "month",
        "day",
        "weekday"
    ]

    tfidf_features = vectorizer.get_feature_names_out()

    all_features = feature_names + list(tfidf_features)

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    feature_df.to_csv(
        os.path.join(
            OUTPUT,
            "feature_importance.csv"
        ),
        index=False
    )
    
    

except Exception as e:

    print("Unable to save feature importance.")
    print(e)

# --------------------------------------------------
# Save Experiment Information
# --------------------------------------------------

with open(
    os.path.join(
        OUTPUT,
        "experiment_info.txt"
    ),
    "w"
) as f:

    f.write("Baseline : Metadata + Notes\n")
    f.write("Model : Random Forest\n")
    f.write(f"Training Samples : {len(y_train)}\n")
    f.write(f"Testing Samples : {len(y_test)}\n")
    f.write(f"Total Features : {X.shape[1]}\n")

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

print("Evaluating...")

metrics = evaluate(
    y_test,
    prediction,
    OUTPUT
)

print("\nEvaluation Summary")

for k, v in metrics.items():

    print(f"{k:<15}: {v:.4f}")

print("\nBaseline 3 Completed Successfully!")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.bar(
    feature_df["Feature"][:20],
    feature_df["Importance"][:20]
)

plt.xticks(rotation=60)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT,
        "feature_importance.png"
    )
)

plt.close()
