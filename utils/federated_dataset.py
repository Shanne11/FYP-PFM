import pandas as pd
import torch


def load_client_dataset(
        csv_path,
        encoders,
        vectorizer
):

    df = pd.read_csv(csv_path)


    from utils.feature_engineering import (
        prepare_metadata_note_features_with_existing
    )


    X, y = prepare_metadata_note_features_with_existing(
        df,
        encoders,
        vectorizer
    )


    X = torch.FloatTensor(
        X.toarray()
    )


    y = torch.LongTensor(
        y.values.copy()
    )


    return X, y