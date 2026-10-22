
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.experiment_data import (
    canonicalize_categories,
    dataset_fingerprint,
)
from utils.proposed_features import ProposedFeatureBuilder


class V2FeatureBuilder(ProposedFeatureBuilder):
    categorical = ProposedFeatureBuilder.categorical + ["currency_marker"]

    def fit(self, train):
        if train["currency_marker"].isna().any():
            raise ValueError("Missing currency marker")

        self.currency_stats = {}
        self.global_stats = self._stats(train["amount"])

        for currency, group in train.groupby("currency_marker"):
            self.currency_stats[currency] = self._stats(group["amount"])

        return super().fit(train)

    @staticmethod
    def _stats(values):
        x = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
        x = np.sign(x) * np.log1p(np.abs(x))
        median = float(np.median(x))
        q25, q75 = np.percentile(x, [25, 75])
        return median, max(float(q75 - q25), 1.0)

    def _prepared(self, frame):
        if "currency_marker" not in frame:
            raise ValueError("V2 requires currency_marker")

        result = super()._prepared(frame)
        result["currency_marker"] = (
            result["currency_marker"].fillna("UNMARKED").astype(str)
        )

        amount = result["amount"].to_numpy(dtype=float)
        amount = np.sign(amount) * np.log1p(np.abs(amount))

        adjusted = []
        for value, currency in zip(amount, result["currency_marker"]):
            median, spread = self.currency_stats.get(
                currency, self.global_stats
            )
            adjusted.append((value - median) / spread)

        result["amount"] = adjusted
        return result


def main():
    data_path = V2 / "data/clean_budgetwise_v2.csv"
    split_path = V2 / "data/experiment_split_v2.json"

    frame = canonicalize_categories(pd.read_csv(data_path))
    split = json.loads(split_path.read_text(encoding="utf-8"))

    assert len(frame) == 14011
    assert dataset_fingerprint(frame) == split["dataset_fingerprint"]

    train = frame.iloc[split["train"]]
    validation = frame.iloc[split["validation"]]

    assert len(train) == 9108
    assert len(validation) == 2102

    builder = V2FeatureBuilder().fit(train)

    train_meta, train_notes, _, train_y = builder.transform_parts(train)
    val_meta, val_notes, _, val_y = builder.transform_parts(validation)

    assert train_meta.shape[1] == val_meta.shape[1]
    assert train_notes.shape[1] == val_notes.shape[1]
    assert np.isfinite(train_meta.data).all()
    assert np.isfinite(val_meta.data).all()
    assert len(set(train_y)) == len(set(val_y)) == 13

    report = {
        "version": "tier_a_v2_only",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows_scored": 0,
        "metadata_features": builder.metadata_size,
        "note_features": builder.note_size,
        "total_features": builder.metadata_size + builder.note_size,
        "categories": len(builder.category_encoder.classes_),
        "currency_handling": "training-fitted per-currency signed-log robust scaling",
        "feature_fitting": "training partition only",
    }

    destination = V2 / "reports/feature_contract_v2.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("V2 feature preparation: PASS")


if __name__ == "__main__":
    main()
