import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_metadata_features(df):
    """
    Prepare features for the Metadata-only baseline.
    """

    df = df.copy()

    # Encode categorical columns
    encoders = {}

    categorical_columns = [
        "transaction_type",
        "payment_mode",
        "location",
        "category"
    ]

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column].astype(str))

        encoders[column] = encoder

    # Features
    X = df[
        [
            "transaction_type",
            "amount",
            "payment_mode",
            "location"
        ]
    ]

    # Label
    y = df["category"]

    return X, y, encoders