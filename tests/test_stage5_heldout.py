import numpy as np
import torch

from experiments.federated_diagnosis.run_stage5_heldout import (
    aggregate_rows,
    evaluate_model,
    ordered_values_sha256,
)


class FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, features):
        return self.fixed_logits[: len(features)]


def test_evaluate_model_reports_fixed_taxonomy_and_zero_recall_categories():
    model = FixedModel([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 3.0, 0.0]])
    features = torch.zeros((3, 2), dtype=torch.float32)
    labels = np.asarray([0, 1, 2])
    metrics, per_class, predictions, distribution = evaluate_model(
        model, features, labels, ["A", "B", "C"]
    )
    assert predictions.tolist() == [0, 1, 1]
    assert metrics["accuracy"] == 2 / 3
    assert metrics["predicted_category_coverage"] == 2
    assert metrics["zero_recall_category_count"] == 1
    assert metrics["zero_recall_categories"] == "C"
    assert per_class.category.tolist() == ["A", "B", "C"]
    assert distribution.predicted_count.tolist() == [1, 2, 0]


def test_test_record_hash_is_order_sensitive():
    assert ordered_values_sha256([1, 2, 3]) == ordered_values_sha256([1, 2, 3])
    assert ordered_values_sha256([1, 2, 3]) != ordered_values_sha256([3, 2, 1])


def test_aggregate_rows_uses_sample_standard_deviation_and_paired_differences():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"seed": 1, "loss": "standard", "accuracy": 0.2, "macro_f1": 0.1,
             "weighted_f1": 0.15, "predicted_category_coverage": 2,
             "zero_recall_category_count": 3},
            {"seed": 2, "loss": "standard", "accuracy": 0.4, "macro_f1": 0.2,
             "weighted_f1": 0.25, "predicted_category_coverage": 3,
             "zero_recall_category_count": 2},
            {"seed": 1, "loss": "effective_number", "accuracy": 0.3, "macro_f1": 0.2,
             "weighted_f1": 0.25, "predicted_category_coverage": 4,
             "zero_recall_category_count": 1},
            {"seed": 2, "loss": "effective_number", "accuracy": 0.5, "macro_f1": 0.3,
             "weighted_f1": 0.35, "predicted_category_coverage": 5,
             "zero_recall_category_count": 0},
        ]
    )
    result = aggregate_rows(frame)
    standard = result[result.comparison == "standard"].iloc[0]
    paired = result[result.comparison == "effective_number_minus_standard_paired"].iloc[0]
    assert np.isclose(standard.accuracy_mean, 0.3)
    assert np.isclose(standard.accuracy_sample_std, np.sqrt(0.02))
    assert np.isclose(paired.accuracy_mean, 0.1)
    assert np.isclose(paired.predicted_category_coverage_mean, 2.0)

