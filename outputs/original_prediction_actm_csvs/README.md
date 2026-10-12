# Original prediction and ACTM CSV package

This package contains byte-for-byte copies of the original experimental outputs used for the Section 16 evidence audit.

## ACTM selection evidence

- `actm_selection_transaction_outcomes.csv`: transaction-level outcomes for seeds 42, 52 and 62 across Random, Entropy-only, Margin-only, Conflict-only and Proposed ACTM.
- `actm_selection_runs.csv`: one result row per seed and strategy.
- `actm_selection_summary.csv`: saved three-seed summary.
- `actm_selection_contract.json`: selection and evaluation contract.

The transaction file has 34,620 rows: 2,308 held-out records x 5 strategies x 3 seeds.

## Matched-ablation predictions

For each seed, `full/predictions.csv` is paired with `without_heuristic_adjustment/predictions.csv`. Each file contains 2,308 rows, and the ordered transaction identifiers match within each seed. Test record IDs, overall metrics and prompt metrics are included beside the predictions.

## Integrity

`MANIFEST.json` records each original path, row count, byte size and SHA-256 hash. The packaging script verifies that copied hashes match the original files before creating the archive.
