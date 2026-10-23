# A3 versus A5 Feature-Budget Comparison

The two tracks answer different engineering questions. Under equal tuning, A5
uses the full fitted character vocabulary in addition to the word branch, so its
actual dimension is substantially larger than A3. Its mean paired A5-A3 accuracy
difference is 3.53%
and its Macro-F1 difference is
0.0155.

Under the matched text-feature budget, A5 divides the available text dimensions
between word and character branches while A3 assigns them to words. Its mean
paired accuracy difference is
-0.08% and its
Macro-F1 difference is
-0.0060. The per-seed signs
must be inspected in the CSV; the evidence does not support a claim that character
features consistently improve classification under a fixed total budget.

All comparisons reuse matched test record IDs. Configuration caps, fitted
dimensions, selected hyperparameters, validation scores and convergence-warning
counts are retained in `A3_A5_FEATURE_BUDGET_COMPARISON.csv`.
