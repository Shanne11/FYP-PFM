# Tier A primary class-weighted confusion recovery audit

Audit date: 2026-10-24 (Asia/Kuala_Lumpur)

## Outcome

Recovered all nine primary class-weighted numerical prediction artefacts from the live FYP-PFM output tree. No retraining, preprocessing refit, threshold change, test-set alteration, inference replay, or image-based estimation was performed.

Overall verification: **PASS**.

## Frozen contract checked

- 105 inputs: 68 metadata features plus 37 Smart Note TF-IDF features.
- 13-category frozen label universe.
- 2,308 frozen held-out test rows from `data/experiment_split.json`.
- 150 simulated clients, 10 communication rounds, three local epochs.
- Inverse-frequency class-weighted cross-entropy.
- Seeds 42, 52 and 62 for FedAvg, FedProx and Proposed.

The FedAvg/FedProx source runner independently enforces the 68 + 37 = 105 input contract. Proposed `model_config.json` records 105 inputs and 13 outputs for every seed.

## Per-run verification

| Seed | Method | Matrix total | Correct | Accuracy | Frozen support match | Prediction SHA-256 |
|---:|---|---:|---:|---:|---|---|
| 42 | fedavg | 2308 | 349 | 0.151213 | True | `bfbbedcf07c78f9b63219a6359c7a26a6dc0fed1a799169d8ff3775a5c7481e1` |
| 42 | fedprox | 2308 | 349 | 0.151213 | True | `0fb0a83259725ffa33ef81a85ac095464e132eb04aed7fa1bf4d089e1d85e21f` |
| 42 | proposed | 2308 | 349 | 0.151213 | True | `9d35115910af742d05b5f4a8de4aee683b8c01369bab83ea2dd3865a00b1c9e8` |
| 52 | fedavg | 2308 | 253 | 0.109619 | True | `1bbde807bcfe86d7cf41d7424b0fbe6151753c03155d90d0078f0086244b325e` |
| 52 | fedprox | 2308 | 260 | 0.112652 | True | `3ea561e09ce4cbdacac5d659f0381fc1f94b572082609f175111322e93fb39c8` |
| 52 | proposed | 2308 | 272 | 0.117851 | True | `42d69091987727e9633cd02cbc3f81449e872e14326a8ec5cb5dbd361eb2e52b` |
| 62 | fedavg | 2308 | 422 | 0.182842 | True | `55eb828dcb011007d91a4f8c79d68e55b6704756cbea4f08ea50ca4e1657a69a` |
| 62 | fedprox | 2308 | 423 | 0.183276 | True | `2c2f0d8204bef95a473f2a274f23fc130d3f543913b3e2b398128f6b13c7241a` |
| 62 | proposed | 2308 | 419 | 0.181542 | True | `d18474c8bb57818d1d751808767dd172d56c81615994d161f4fe543a9af4a46e` |

## Top actual-to-predicted off-diagonal pairs

The CSV contains the top ten per run; the table below shows the top five per run.

| Seed | Method | Rank | Actual | Predicted | Count |
|---:|---|---:|---|---|---:|
| 42 | fedavg | 1 | Food | Travel | 383 |
| 42 | fedavg | 2 | Rent | Travel | 296 |
| 42 | fedavg | 3 | Utilities | Travel | 178 |
| 42 | fedavg | 4 | Entertainment | Travel | 153 |
| 42 | fedavg | 5 | Education | Travel | 114 |
| 42 | fedprox | 1 | Food | Travel | 390 |
| 42 | fedprox | 2 | Rent | Travel | 302 |
| 42 | fedprox | 3 | Utilities | Travel | 181 |
| 42 | fedprox | 4 | Entertainment | Travel | 159 |
| 42 | fedprox | 5 | Education | Travel | 115 |
| 42 | proposed | 1 | Food | Travel | 388 |
| 42 | proposed | 2 | Rent | Travel | 308 |
| 42 | proposed | 3 | Utilities | Travel | 183 |
| 42 | proposed | 4 | Entertainment | Travel | 156 |
| 42 | proposed | 5 | Education | Travel | 117 |
| 52 | fedavg | 1 | Food | Utilities | 481 |
| 52 | fedavg | 2 | Rent | Utilities | 380 |
| 52 | fedavg | 3 | Travel | Utilities | 282 |
| 52 | fedavg | 4 | Entertainment | Utilities | 189 |
| 52 | fedavg | 5 | Education | Utilities | 140 |
| 52 | fedprox | 1 | Food | Utilities | 477 |
| 52 | fedprox | 2 | Rent | Utilities | 377 |
| 52 | fedprox | 3 | Travel | Utilities | 280 |
| 52 | fedprox | 4 | Entertainment | Utilities | 187 |
| 52 | fedprox | 5 | Education | Utilities | 139 |
| 52 | proposed | 1 | Food | Utilities | 477 |
| 52 | proposed | 2 | Rent | Utilities | 378 |
| 52 | proposed | 3 | Travel | Utilities | 281 |
| 52 | proposed | 4 | Entertainment | Utilities | 190 |
| 52 | proposed | 5 | Education | Utilities | 140 |
| 62 | fedavg | 1 | Food | Travel | 276 |
| 62 | fedavg | 2 | Rent | Travel | 217 |
| 62 | fedavg | 3 | Entertainment | Travel | 119 |
| 62 | fedavg | 4 | Utilities | Travel | 117 |
| 62 | fedavg | 5 | Food | Rent | 78 |
| 62 | fedprox | 1 | Food | Travel | 276 |
| 62 | fedprox | 2 | Rent | Travel | 217 |
| 62 | fedprox | 3 | Entertainment | Travel | 119 |
| 62 | fedprox | 4 | Utilities | Travel | 117 |
| 62 | fedprox | 5 | Food | Rent | 75 |
| 62 | proposed | 1 | Food | Travel | 263 |
| 62 | proposed | 2 | Rent | Travel | 216 |
| 62 | proposed | 3 | Utilities | Travel | 115 |
| 62 | proposed | 4 | Entertainment | Travel | 112 |
| 62 | proposed | 5 | Food | Rent | 74 |

## Provenance and limitations

- Three local FYP-PFM Git worktrees/copies were located and searched at commits `2643e425...`, `c421ee22...` and `aa7082d0...`. The nine requested numerical artefacts exist only in the current repository copy. Absolute workstation paths are retained only in the controlled local audit.
- The numerical output folders are not listed by `git ls-files`; their authenticity is therefore tied to their SHA-256 hashes, frozen split alignment, internally consistent metrics/configuration and the local evidence trail, rather than direct Git blob identity.
- `git status` returned `fatal: this operation must be run in a work tree` despite `git rev-parse --is-inside-work-tree` returning `true`; no Git configuration was changed.
- Proposed seeds 52 and 62 contain a stale nested `split.seed` value of 42. This is consistent with the one intentionally frozen seed-42 split manifest used across training seeds; `training_seed` correctly records 52 and 62. The audit reports this metadata defect without modifying it.
- `FYP-PFM_TierA_V2_Freeze_20261023.zip` is a separate Tier A V2 validation freeze and does not contain these nine primary class-weighted prediction files. `FYP-PFM10102026.zip` contains class-balance summaries but not the nine prediction-level artefacts.
- Exact matrices were reconstructed from CSV labels only. PNG confusion matrices were not read or used to estimate counts.
- This recovery audit does not change any Chapter 5 number or RQ/RO verdict.

## Audit files

- `confusion_matrices/`: nine exact 13x13 matrices.
- `top_10_confusion_pairs_per_run.csv`: ranked leading off-diagonal pairs.
- `all_off_diagonal_confusion_pairs.csv`: every non-zero off-diagonal pair.
- `frozen_test_support.csv` and `per_category_support_checks.csv`: reference and per-run support verification.
- `verification_checks.csv`: all row, mapping, split, support, metric and configuration checks.
- `source_artifacts.csv`: repository-relative source paths, sizes and SHA-256 hashes. Absolute paths were removed from this public copy.
- The detailed local search inventory remains in controlled local storage because it contains workstation paths.
- `recover_confusion_audit.ps1`: reproducible read-only audit script.
