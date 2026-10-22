# Stage 5 reproduction

Run from the repository root using the existing pinned environment:

```powershell
.\.venv\Scripts\python.exe experiments\federated_diagnosis\run_stage5_heldout.py
.\.venv\Scripts\python.exe -m pytest tests\test_stage5_heldout.py tests\test_class_balance_screen.py tests\test_class_balance_replication.py -q
```

The output directory must be new or empty. The script verifies the Stage 3 and Stage 4 manifests, replays the fixed six training runs, requires exact Stage 4 validation-trajectory reproduction within `1e-12`, and only then evaluates the held-out test partition. Do not use Stage 5 results to tune the configuration.
