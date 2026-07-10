import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from utils.feature_engineering import prepare_metadata_features
from utils.metrics import evaluate

from models.metadata_model import build_model

OUTPUT = "outputs/baseline2"

os.makedirs(OUTPUT, exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/clean_budgetwise.csv")

# Feature engineering
X, y, encoders = prepare_metadata_features(df)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_index = X_train.index
test_index = X_test.index

pd.Series(train_index).to_csv(
    os.path.join(
        OUTPUT,
        "train_indices.csv"
    ),
    index=False,
    header=False
)

pd.Series(test_index).to_csv(
    os.path.join(
        OUTPUT,
        "test_indices.csv"
    ),
    index=False,
    header=False
)

with open(
    os.path.join(
        OUTPUT,
        "experiment_info.txt"
    ),
    "w"
) as f:

    f.write(f"Training Samples : {len(X_train)}\n")

    f.write(f"Testing Samples  : {len(X_test)}\n")

    f.write(f"Features         : {list(X.columns)}\n")

    f.write("Model            : Random Forest\n")

# Build model
model = build_model()

# Train
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

importance = model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    "Feature": feature_names,
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

metrics = evaluate(
    y_test,
    prediction,
    OUTPUT
)

# Save model
joblib.dump(
    model,
    os.path.join(
        OUTPUT,
        "metadata_model.pkl"
    )
)

joblib.dump(
    encoders,
    os.path.join(
        OUTPUT,
        "label_encoders.pkl"
    )
)

# Save predictions
results = X_test.copy()

results["Actual"] = y_test.values

results["Predicted"] = prediction

results.to_csv(
    os.path.join(
        OUTPUT,
        "predictions.csv"
    ),
    index=False
)

print(metrics)

print("\nMetadata baseline completed.")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save indices for reuse
X_train.index.to_series().to_csv(
    os.path.join(OUTPUT, "train_indices.csv"),
    index=False
)

X_test.index.to_series().to_csv(
    os.path.join(OUTPUT, "test_indices.csv"),
    index=False
)

category_encoder = encoders["category"]

mapping = {
    label: int(index)
    for index, label in enumerate(category_encoder.classes_)
}

pd.DataFrame(
    mapping.items(),
    columns=["Category", "Encoded"]
).to_csv(
    os.path.join(
        OUTPUT,
        "category_mapping.csv"
    ),
    index=False
)

importance = model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(
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

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT,
        "feature_importance.png"
    )
)

plt.close()