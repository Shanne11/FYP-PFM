# Reproduction

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\federated_diagnosis\run_class_balance_replication.py --seeds 52,62
```

Seed 42 is reused only after verifying all Stage 3 manifest hashes and its configuration. The script does not transform, predict, select, or score the held-out test partition.

Focused regression tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_class_balance_replication.py tests\test_class_balance_screen.py tests\test_optimisation_screen.py -q
```

On the execution host, the repository `.venv` launcher currently points to a removed Windows Store Python. The focused tests were therefore run with the already-provisioned compatible project runtime at `C:\Users\thepr\OneDrive\Documents\ChatGPT\FYP\github_readme_updates\FYP-PFM\.venv\Scripts\python.exe` (Python 3.11; PyTorch 2.14.0+cpu, pandas 2.3.3, scikit-learn 1.9.1 and pytest 8.4.2). No dependency or source file was changed to work around this host-specific launcher issue.
