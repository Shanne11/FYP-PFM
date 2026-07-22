import numpy as np

from evaluation.paired_statistics import exact_sign_flip_pvalue, paired_row


def test_three_consistent_differences_have_minimum_two_sided_exact_p_value():
    assert exact_sign_flip_pvalue([1.0, 1.0, 1.0]) == 0.25


def test_paired_row_counts_directional_wins():
    row = paired_row("example", "macro_f1", [0.3, 0.4, 0.5], [0.2, 0.3, 0.4], "a", "b")
    assert row["wins_a"] == 3
    assert np.isclose(row["mean_difference_a_minus_b"], 0.1)
