import pandas as pd
import pytest

from utils.proposed_features import ProposedFeatureBuilder


def frame(**overrides):
    values = {
        "transaction_type": ["Expense", "Expense"],
        "payment_mode": ["Card", "Cash"],
        "location": ["KL", "PJ"],
        "amount": [10.0, 20.0],
        "date": ["2026-01-01", "2026-01-02"],
        "notes": ["meal", "purchase"],
        "category": ["Food", "Other"],
        "merchant": ["Mamak Corner", "AEON Mall"],
        "description": ["nasi lemak", "household purchase"],
    }
    values.update(overrides)
    return pd.DataFrame(values)


def test_original_contract_does_not_require_transaction_text():
    builder = ProposedFeatureBuilder().fit(
        frame().drop(columns=["merchant", "description"])
    )
    assert builder.transaction_text_vectorizer is None


def test_merchant_aware_contract_requires_real_columns():
    builder = ProposedFeatureBuilder(
        transaction_text_columns=["merchant", "description"],
        max_transaction_text_features=20,
    )
    with pytest.raises(ValueError, match="missing: merchant, description"):
        builder.fit(frame().drop(columns=["merchant", "description"]))


def test_transaction_text_vocabulary_is_fitted_on_training_rows_only():
    builder = ProposedFeatureBuilder(
        transaction_text_columns=["merchant", "description"],
        max_transaction_text_features=20,
    ).fit(frame())
    vocabulary = builder.transaction_text_vectorizer.vocabulary_
    assert "mamak" in vocabulary
    assert "futureunseenmerchant" not in vocabulary

    test = frame(
        merchant=["FutureUnseenMerchant", "AEON Mall"],
        description=["unseenword", "household purchase"],
    )
    metadata, _, _, _ = builder.transform_parts(test)
    assert metadata.shape[1] == builder.metadata_size
    assert "futureunseenmerchant" not in builder.transaction_text_vectorizer.vocabulary_
