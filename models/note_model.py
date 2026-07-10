from sklearn.ensemble import RandomForestClassifier


def build_model():

    return RandomForestClassifier(

        n_estimators=200,

        max_depth=15,

        min_samples_split=5,

        random_state=42,

        n_jobs=-1

    )