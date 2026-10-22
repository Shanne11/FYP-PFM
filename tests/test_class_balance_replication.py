import pandas as pd

from experiments.federated_diagnosis.run_class_balance_replication import seed_summary


def test_seed_summary_uses_fixed_round_25_and_separate_peak():
    rounds = pd.DataFrame([
        {"round": 1, "validation_accuracy": 0.2, "validation_macro_f1": 0.3,
         "validation_weighted_f1": 0.25, "predicted_class_coverage": 4,
         "single_class_collapse": False},
        {"round": 16, "validation_accuracy": 0.3, "validation_macro_f1": 0.2,
         "validation_weighted_f1": 0.28, "predicted_class_coverage": 3,
         "single_class_collapse": False},
        {"round": 25, "validation_accuracy": 0.4, "validation_macro_f1": 0.1,
         "validation_weighted_f1": 0.3, "predicted_class_coverage": 2,
         "single_class_collapse": False},
    ])
    classes = pd.DataFrame([
        {"round": 25, "recall": 0.0}, {"round": 25, "recall": 0.5}
    ])
    result = seed_summary(rounds, classes, "effective_number", 42)
    assert result["peak_round"] == 1
    assert result["round_25_macro_f1"] == 0.1
    assert result["round_25_zero_recall_categories"] == 1
