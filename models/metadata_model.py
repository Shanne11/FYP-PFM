import pandas as pd

# from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import LabelEncoder

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier


def build_model():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    return model
