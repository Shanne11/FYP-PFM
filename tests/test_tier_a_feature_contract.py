import pandas as pd
import pytest

from utils.proposed_features import ProposedFeatureBuilder


def frame():
    return pd.DataFrame({
        "transaction_type": ["Expense", "Expense"],
        "payment_mode": ["Card", "Cash"],
        "location": ["KL", "PJ"],
        "amount": [10.0, 20.0],
        "date": ["2026-01-01", "2026-01-02"],
        "notes": ["meal", "purchase"],
        "category": ["Food", "Other"],
    })


def test_tier_a_contract_does_not_require_merchant_or_description():
    builder = ProposedFeatureBuilder().fit(frame())
    metadata, notes, anchors, labels = builder.transform_parts(frame())
    assert metadata.shape[1] == builder.metadata_size
    assert notes.shape == anchors.shape
    assert len(labels) == len(frame())
    assert not hasattr(builder, "transaction_text_vectorizer")


def test_tier_a_builder_rejects_removed_deployment_text_options():
    with pytest.raises(TypeError):
        ProposedFeatureBuilder(
            transaction_text_columns=["merchant", "description"],
            max_transaction_text_features=20,
        )
