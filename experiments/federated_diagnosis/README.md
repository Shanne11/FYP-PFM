# Federated Diagnosis Experiment

This directory contains the initial validation-only diagnosis, optimisation screens, three-seed replication and the subsequently authorised Stage 5 held-out extension. It does not modify the historical Section 16 evidence or PocketIQ.

## Reproduce the completed diagnosis

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe experiments\federated_diagnosis\run_diagnosis.py --epochs 30 --seeds 42,52,62
```

The script fits all preprocessing on training data, audits all 150 clients, trains lightweight centralised controls, and copies historical federated summaries into this isolated directory. Newly trained models are evaluated on validation data only.

## Outputs

- `BASELINE_AUDIT.md` — repository, contract, and source audit.
- `FEDERATED_UNDERFITTING_DIAGNOSIS.md` — evidence-backed diagnosis.
- `CLIENT_DISTRIBUTION_ANALYSIS.md` — simulated-client heterogeneity.
- `PROPOSED_TRAINING_EXPERIMENTS.md` — staged plan awaiting approval.
- `outputs/` — machine-readable CSV/JSON evidence.

Large sweeps and held-out evaluation require explicit approval after review of this diagnosis.

## Approved Stage 1 optimisation screen

The seed-42 reset-Adam versus reset-SGD screen can be reproduced with:

```powershell
.\.venv\Scripts\python.exe experiments\federated_diagnosis\run_optimisation_screen.py
```

Results are stored only in `outputs/optimisation_screen/`. SGD briefly improved early-round Macro F1 but collapsed to a single predicted category by round 25, so conditional replication on seeds 52 and 62 was not performed. See `outputs/optimisation_screen/OPTIMISATION_SCREEN_RESULTS.md`.

## Adam-state persistence experiment

The seed-42 reset-Adam versus private per-client persistent-Adam comparison can be reproduced with:

```powershell
.\.venv\Scripts\python.exe experiments\federated_diagnosis\run_adam_persistence.py
```

Persistent Adam did not materially improve classification stability. Both variants entered single-class collapse and finished with two-class coverage. Results and hashes are under `outputs/adam_persistence/`.

## Class-balanced loss screen

The seed-42 standard versus effective-number loss comparison can be reproduced with:

```powershell
.\.venv\Scripts\python.exe experiments\federated_diagnosis\run_class_balance_screen.py
```

Effective-number loss prevented single-class collapse for seed 42 and improved both validation accuracy and Macro F1 through round 25. Results are under `outputs/class_balance_screen/`; replication has been completed for seeds 42, 52 and 62.

## Completed Stage 4 replication and Stage 5 held-out evaluation

- [Stage 4 report](outputs/class_balance_replication/THREE_SEED_REPLICATION_RESULTS.md): validation accuracy 0.2344 +/- 0.0052 and Macro F1 0.0834 +/- 0.0111 for effective-number loss.
- [Stage 5 report](outputs/stage5_heldout/STAGE5_HELDOUT_RESULTS.md): held-out accuracy 0.2327 +/- 0.0070 and Macro F1 0.0831 +/- 0.0087, versus standard CE 0.2202 +/- 0.0051 and 0.0365 +/- 0.0082.

Both stages use 68 metadata features, 25 rounds and fixed round-25 comparisons. These are separate exploratory experiments, not the primary 105-feature, 10-round Tier A Proposed evaluation. The held-out partition had been inspected previously.

Reproduction entry points are `run_class_balance_replication.py` and `run_stage5_heldout.py`. See their output directories for configurations and historical manifests. Some CSVs/checkpoints referenced by those manifests are not tracked in this repository; the reports alone do not constitute a complete artifact bundle. No fresh training was run for this documentation update.
