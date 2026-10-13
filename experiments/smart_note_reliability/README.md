# Smart Note reliability experimental extension

This directory is separate from the verified Section 16 evidence. It evaluates effective-number class-balanced loss and Reliability-Gated Smart Note Fusion (RGSF) without changing the PocketIQ deployment model.

## Method

For training-class count \(n_c\), the effective-number loss weight is

\[
w_c = \frac{1-\beta}{1-\beta^{n_c}}, \qquad \beta=0.9999,
\]

followed by mean-one normalisation. Counts are calculated from the frozen training partition only.

For each ACTM-selected record with an available retrospective source note, RGSF compares same-checkpoint base and note-enhanced distributions. The transparent gate accepts a changed note prediction only if all validation-selected thresholds are met:

\[
\Delta confidence \ge t_c,\quad
\Delta margin \ge t_m,\quad
\Delta entropy \ge t_h,\quad
novelty \ge t_n.
\]

The transaction decision uses no true label. Validation labels select the threshold tuple by Macro F1, followed by accuracy, net corrections, fewer confident-wrong cases and lower coverage. The held-out labels are used only for final evaluation.

## Reproduction

From the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_smart_note_reliability.py tests\test_class_balance.py tests\test_tier_a_feature_contract.py tests\test_experiment_data.py -q
.\.venv\Scripts\python.exe experiments\smart_note_reliability\run_study.py --seeds 42 52 62 --rounds 10 --local-epochs 3 --effective-beta 0.9999 --output experiments\smart_note_reliability\results
```

To regenerate tables without retraining completed extension checkpoints:

```powershell
.\.venv\Scripts\python.exe experiments\smart_note_reliability\run_study.py --seeds 42 52 62 --rounds 10 --local-epochs 3 --effective-beta 0.9999 --reuse --output experiments\smart_note_reliability\results
```

Do not use `--max-clients` for reportable evidence; it is a smoke-test control only.

## Evidence boundary

BudgetWise `notes` are retrospective source-note proxies. They are not participant-entered clarifications, and the results do not establish real-user usefulness. Full checkpoints and transaction-level CSVs remain local; compact reviewed tables and hashes may be versioned.
