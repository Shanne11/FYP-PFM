# Tier A primary class-weighted confusion recovery audit

This directory preserves non-sensitive aggregate evidence reconstructed from the frozen Tier A numerical predictions for FedAvg, FedProx and Proposed, using seeds 42, 52 and 62.

The audit covers the frozen 105-feature, 13-category and 2,308-record held-out configuration. Every matrix totals 2,308, and every per-category actual support matches the frozen held-out labels.

## Privacy boundary

This public evidence does not contain original prediction-level records, transaction identifiers, transaction descriptions, Smart Notes, raw datasets, fitted preprocessing objects or model checkpoints. Those artefacts remain in controlled local storage.

`source_artifacts.csv` is the public version of the provenance index. It contains repository-relative paths and SHA-256 hashes only; absolute local filesystem paths were removed before publication.

## Contents

- `TIER_A_PRIMARY_CONFUSION_RECOVERY_AUDIT.md`: audit findings and limitations.
- `verification_checks.csv`: per-run configuration, alignment, support and metric checks.
- `top_10_confusion_pairs_per_run.csv`: leading actual-to-predicted off-diagonal pairs.
- `all_off_diagonal_confusion_pairs.csv`: all non-zero aggregate confusion pairs.
- `confusion_matrices/`: nine reconstructed 13x13 aggregate matrices.
- `frozen_test_support.csv`: frozen held-out support for all 13 categories.
- `per_category_support_checks.csv`: per-run support verification.
- `source_artifacts.csv`: privacy-safe source paths, sizes and hashes.
- `audit_artifact_manifest.csv`: hashes for the published audit files.

No model was retrained and no Chapter 5 RQ/RO verdict was changed by this audit.
