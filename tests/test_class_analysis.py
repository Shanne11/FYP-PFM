import pandas as pd

from evaluation.analyze_federated_classes import class_rows, summarize_classes


def test_class_analysis_detects_unpredicted_category():
    frame = pd.DataFrame({"actual": ["A", "A", "B"], "predicted": ["A", "A", "A"]})
    rows = class_rows(frame, "fedavg", 42)
    summary = summarize_classes(pd.DataFrame(rows)).set_index("category")
    assert summary.loc["B", "recall_mean"] == 0
    assert summary.loc["B", "never_predicted_seeds"] == 1
    assert summary.loc["A", "recall_mean"] == 1
