# Missing Class-Weighted Tier A Experiments — Verified Results

All six authorised runs completed using the frozen 10-round Tier A configuration. Metrics were independently recomputed from the saved held-out probability matrices; record order, labels, category mappings and intervention settings passed verification.

| Experiment | Proposed Macro F1 | Comparator Macro F1 | Difference | 95% paired t interval | Wins | Verdict |
|---|---:|---:|---:|---:|---:|---|
| RQ3: Proposed vs class-weighted A1 | 0.066090 | 0.066504 | -0.000413 | [-0.001273, 0.000446] | 0/3 | Not Achieved |
| RQ4: Proposed vs class-weighted no-adjustment | 0.066090 | 0.066795 | -0.000705 | [-0.003723, 0.002313] | 0/3 | Not Achieved |

## Scientific interpretation

- RQ3/RO3 is decided only by the matched class-weighted Proposed-versus-A1 comparison. Expanded V2 remains separate supporting centralised genuine-note evidence.
- RQ4/RO5 is decided only by the matched class-weighted Proposed-versus-no-adjustment comparison. Continuous aggregation coefficients may differ even when categorical outcomes are nearly or exactly unchanged.
- Three seeds provide descriptive paired uncertainty, not proof of population-level statistical significance.

## Provenance limitation

The saved class-weighted Proposed outputs predate the explicit `loss_mode` metadata field. They record `class_weighted_loss=true`, and their saved 13-value class-weight vector exactly matches inverse-frequency weights recomputed from the frozen training partition. They were therefore accepted as matched comparators without retraining.

