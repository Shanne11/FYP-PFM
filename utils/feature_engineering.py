import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_metadata_features(df):

    df = df.copy()

    # ---------- Amount ----------
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    # ---------- Date ----------
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=False
    )

    df["year"] = df["date"].dt.year.fillna(0).astype(int)
    df["month"] = df["date"].dt.month.fillna(0).astype(int)
    df["day"] = df["date"].dt.day.fillna(0).astype(int)
    df["weekday"] = df["date"].dt.weekday.fillna(0).astype(int)

    # ---------- Missing values ----------
    df["transaction_type"] = df["transaction_type"].fillna("Unknown")
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df["location"] = df["location"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")

    encoders = {}

    categorical_columns = [
        "transaction_type",
        "payment_mode",
        "location",
        "category"
    ]

    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        encoders[col] = encoder

    X = df[
        [
            "transaction_type",
            "amount",
            "payment_mode",
            "location",
            "year",
            "month",
            "day",
            "weekday"
        ]
    ]

    y = df["category"]

    return X, y, encoders