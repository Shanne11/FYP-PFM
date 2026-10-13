# Test execution record

Environment: Python 3.11.9, PyTorch 2.13.0+cpu, NumPy 2.4.6, pandas 2.3.3 and scikit-learn 1.9.0.

## Targeted safeguards

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_smart_note_reliability.py tests\test_class_balance.py tests\test_tier_a_feature_contract.py tests\test_experiment_data.py -q
```

Result: **13 passed** in 44.67 seconds.

## Full tracked Tier A suite plus extension

Command:

```powershell
$tracked = git ls-files "tests/*.py"
.\.venv\Scripts\python.exe -m pytest --import-mode=importlib @tracked tests\test_smart_note_reliability.py -q
```

Final non-duplicated result: **44 passed** in 5.66 seconds.

## Default discovery limitation

Plain `pytest -q` does not collect successfully in this local checkout because the user-owned untracked nested `FYP-PFM/` repository contains duplicate test module names, and the separate untracked `tests/test_frozen_evaluation_contracts.py` imports `matched_random_selection`, which is absent from the tracked ACTM source. Those unrelated untracked files were not modified. This is a collection-environment limitation, not a failure in the 44 tracked-plus-extension tests.

## Artifact verification

The generated manifest declares 18 compact result artifacts. Independent SHA-256 recalculation verified all **18/18**, with no missing or mismatched file. The three transaction-level files each contain 2,308 unique held-out source rows.
