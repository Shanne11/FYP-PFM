"""Leakage-safe feature preparation for the proposed experiment."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class ProposedFeatureBuilder:
    categorical = ["transaction_type", "payment_mode", "location"]
    numeric = ["amount", "year", "month", "day", "weekday"]

    def __init__(
        self,
        max_note_features=500,
        transaction_text_columns=None,
        max_transaction_text_features=0,
    ):
        self.transaction_text_columns = list(transaction_text_columns or [])
        self.max_transaction_text_features = int(max_transaction_text_features)
        if self.max_transaction_text_features < 0:
            raise ValueError("max_transaction_text_features cannot be negative")
        if bool(self.transaction_text_columns) != bool(self.max_transaction_text_features):
            raise ValueError(
                "transaction_text_columns and max_transaction_text_features must be enabled together"
            )
        self.category_encoder = LabelEncoder()
        self.categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.numeric_scaler = StandardScaler()
        self.note_vectorizer = TfidfVectorizer(
            max_features=max_note_features, lowercase=True, stop_words="english"
        )
        self.transaction_text_vectorizer = (
            TfidfVectorizer(
                max_features=self.max_transaction_text_features,
                lowercase=True,
                stop_words="english",
            )
            if self.transaction_text_columns
            else None
        )

    def _transaction_text(self, frame):
        if not self.transaction_text_columns:
            return None
        missing = [column for column in self.transaction_text_columns if column not in frame]
        if missing:
            raise ValueError(
                "Merchant-aware features require genuine labelled columns; missing: "
                + ", ".join(missing)
            )
        values = frame[self.transaction_text_columns].fillna("").astype(str)
        combined = values.agg(" ".join, axis=1).str.strip()
        if not combined.ne("").any():
            raise ValueError("Merchant-aware text columns contain no usable training text")
        return combined

    @staticmethod
    def _prepared(frame):
        frame = frame.copy()
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
        dates = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
        frame["year"] = dates.dt.year.fillna(0)
        frame["month"] = dates.dt.month.fillna(0)
        frame["day"] = dates.dt.day.fillna(0)
        frame["weekday"] = dates.dt.weekday.fillna(0)
        for column in ProposedFeatureBuilder.categorical:
            frame[column] = frame[column].fillna("Unknown").astype(str)
        frame["notes"] = frame["notes"].fillna("").astype(str)
        return frame

    def fit(self, train_frame):
        frame = self._prepared(train_frame)
        self.category_encoder.fit(frame["category"].fillna("Unknown").astype(str))
        self.categorical_encoder.fit(frame[self.categorical])
        self.numeric_scaler.fit(frame[self.numeric])
        notes = frame["notes"]
        # Ensure a vocabulary exists even if a tiny development split has no notes.
        self.note_vectorizer.fit(pd.concat([notes, pd.Series(["unknown note"])], ignore_index=True))
        transaction_text = self._transaction_text(train_frame)
        if transaction_text is not None:
            # Fitted on the training split only. Validation and test text can
            # transform through this frozen vocabulary but never influence it.
            self.transaction_text_vectorizer.fit(
                pd.concat(
                    [transaction_text, pd.Series(["unknown merchant description"])],
                    ignore_index=True,
                )
            )
        return self

    def transform_parts(self, frame):
        prepared = self._prepared(frame)
        categorical = self.categorical_encoder.transform(prepared[self.categorical])
        numeric = csr_matrix(self.numeric_scaler.transform(prepared[self.numeric]))
        metadata_parts = [categorical, numeric]
        transaction_text = self._transaction_text(frame)
        if transaction_text is not None:
            metadata_parts.append(self.transaction_text_vectorizer.transform(transaction_text))
        metadata = hstack(metadata_parts, format="csr")
        notes = self.note_vectorizer.transform(prepared["notes"])
        anchors = self.note_vectorizer.transform(
            prepared[["transaction_type", "payment_mode", "location"]].agg(" ".join, axis=1)
        )
        labels = self.category_encoder.transform(
            prepared["category"].fillna("Unknown").astype(str)
        )
        return metadata, notes, anchors, labels

    @property
    def metadata_size(self):
        text_size = (
            len(self.transaction_text_vectorizer.get_feature_names_out())
            if self.transaction_text_vectorizer is not None
            and hasattr(self.transaction_text_vectorizer, "vocabulary_")
            else 0
        )
        return len(self.categorical_encoder.get_feature_names_out()) + len(self.numeric) + text_size

    @property
    def metadata_feature_names(self):
        names = list(self.categorical_encoder.get_feature_names_out()) + [
            f"numeric_scaled__{column}" for column in self.numeric
        ]
        if self.transaction_text_vectorizer is not None:
            names.extend(
                f"transaction_text_tfidf__{token}"
                for token in self.transaction_text_vectorizer.get_feature_names_out()
            )
        return names

    @property
    def note_size(self):
        return len(self.note_vectorizer.get_feature_names_out())

    def fused(self, metadata, notes, use_note_mask):
        masked = notes.multiply(np.asarray(use_note_mask, dtype=float)[:, None])
        return hstack([metadata, masked], format="csr")

    def semantic_anchor_fused(self, metadata, notes, anchors, use_note_mask):
        """Gate note features by information added beyond their context anchor."""
        specificity = np.asarray([
            1.0 - cosine_similarity(notes[index], anchors[index])[0, 0]
            if notes[index].nnz and anchors[index].nnz else 0.0
            for index in range(notes.shape[0])
        ])
        # Preserve useful note signal while increasing influence for specific notes.
        gates = (0.5 + 0.5 * np.clip(specificity, 0.0, 1.0))
        gates *= np.asarray(use_note_mask, dtype=float)
        return hstack([metadata, notes.multiply(gates[:, None])], format="csr")

    def metadata_only(self, metadata):
        return hstack([metadata, csr_matrix((metadata.shape[0], self.note_size))], format="csr")
