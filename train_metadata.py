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

# Build model
model = build_model()

# Train
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

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