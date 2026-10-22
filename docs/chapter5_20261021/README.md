# Verified primary class-weighted Tier A comparisons

Published documentation update: 22 October 2026.

This directory publishes the completed 21 October six-run RQ3/RQ4 summaries and original verification records. It does not rerun training or replace the frozen Section 16 archive.

## Evidence
- [Comparison summary](CLASS_WEIGHTED_COMPARISON_SUMMARY.csv)
- [Paired seed results](CLASS_WEIGHTED_PAIRED_SEED_RESULTS.csv)
- [Scientific interpretation](MISSING_CLASS_WEIGHTED_EXPERIMENT_REPORT.md)
- [Independent run verification](INDEPENDENT_RUN_VERIFICATION.csv)
- [Original pre-run test record](pre_run_tests.xml)
- [Original post-run test record](post_run_tests.xml)

Both historical XML records report 13 passing tests. These are the October 21 execution records, not newly executed tests.

## Frozen comparison
105 features, 13 categories, 150 training clients, seeds 42/52/62, 10 communication rounds, three local epochs, batch size 32, learning rate 0.001, training-only inverse-frequency class weighting, and matched aggregation except for the intended intervention.

RQ3 disables the 37-feature Smart Note block. RQ4 disables the clarification-derived aggregation adjustment. Existing Proposed checkpoints were reused. Their metadata predates the explicit loss_mode field, but records class_weighted_loss=true and a matching training-derived class-weight vector.

RQ3 mean Macro F1 difference is -0.00041329290437296107; RQ4 is -0.0007049878855238181. Both are Not Achieved under the frozen criteria. Expanded V2 remains separate evidence.

## Reproducibility boundary
This is a published summary subset of the local CHAPTER5_MISSING_TIER_A_20261021 evidence bundle. Full checkpoints, probabilities, runtime source snapshots and execution logs are not included here. Independent numerical reproduction from this GitHub directory alone is therefore incomplete; obtain the original frozen bundle for prediction-level verification.

## Original local source SHA-256
These identify original local files read during publication; they are not a new verification of the complete historical bundle.

| File | Original SHA-256 |
|---|---|
| MISSING_CLASS_WEIGHTED_EXPERIMENT_REPORT.md | FA7C1BA8B771AE514514D40360A1FA4F5B92148F495A5979531028660ADA8F05 |
| CLASS_WEIGHTED_COMPARISON_SUMMARY.csv | 6ED8141582266CECC763AFFFAE46290525CAA52F43C886E964A560D9F6BD043B |
| CLASS_WEIGHTED_PAIRED_SEED_RESULTS.csv | 797FD06E340067165E94594068D49FB5758C41180A352DF2A20C0260762EF3F3 |
| INDEPENDENT_RUN_VERIFICATION.csv | FD81E4C18F2DBF9CCB73CF5586D413708AFA2770794604DAB20E9390B647EB93 |
| pre_run_tests.xml | 4A76AFC2820E801EA4F8DE66BD383D899E1BF7D403857E5ADA14D9F9CE44468F |
| post_run_tests.xml | C3AFFD5B5639CC561DF78A5FFBC99EEE0D2195992DD6AA975810BE35718D966A |

## Current publication checks
The dashboard's two existing regression tests passed on 22 October 2026 using Python 3.11 and the existing project site-packages. The old virtual-environment launcher failed because its base Python path no longer exists; using the installed Python interpreter resolved execution. No models were retrained.

