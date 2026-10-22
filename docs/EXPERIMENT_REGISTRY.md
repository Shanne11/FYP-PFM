# FYP-PFM Experiment Registry

Last documented: 22 October 2026.

This registry separates the primary thesis experiments
from supplementary and exploratory research. It does
not change experimental protocols or research results.

## Research pathways

| Pathway | Contract | Research role |
| --- | --- | --- |
| Primary Tier A | 105 features; 13 categories; 150 clients; 10 rounds; seeds 42, 52, 62 | Authoritative federated evaluation for thesis RQs |
| BudgetWise Tier A V2 | 108 features; 13 categories; 150 clients; 40 rounds; seed 42 | Exploratory validation and Smart Note diagnostics |
| Federated diagnosis | Separate optimisation and class-balance configurations | Supporting investigation of underfitting |
| Expanded V2 Stage A/B/C | Separate description and genuine owner-written note evaluation | Complementary centralised clarification evidence |
| PocketIQ deployment | Separate 4,096-feature, 14-category ONNX package | Mobile deployment and functional verification |

## Primary Tier A: authoritative evidence

The frozen primary class-weighted evaluation uses:

- 68 metadata features and 37 Smart Note features.
- 150 simulated training clients.
- 10 communication rounds and 3 local epochs.
- Seeds 42, 52 and 62.
- Training-derived inverse-frequency class weighting.

The completed matched comparisons did not demonstrate
a Macro F1 improvement from the Smart Note block
or bounded heuristic-adjusted aggregation.

Authoritative indexes:

- [Primary research overview](../README.md)
- [Section 16 evidence](section16/README.md)
- [Class-weighted comparison verification](chapter5_20261021/README.md)

## BudgetWise Tier A V2: exploratory extension

The separate V2 experiment uses 108 features, 40 rounds,
effective-number loss and a frozen transaction-ID group-safe
split of 9,108 train, 2,102 validation and 2,801 test records.

Published V2 evaluations and Smart Note diagnostics use
validation data only. Their summaries report zero
held-out test rows scored.

Results identify weak alignment between retrospective
source notes and transaction categories.

- [V2 experiment guide](../experiments/tier_a_v2/README.md)
- [V2 training runner](../experiments/tier_a_v2/run_validation_v2.py)

These V2 results must not replace the primary Tier A
105-feature research protocol.

## Federated diagnosis extension

The underfitting and optimisation experiments have
their own configurations and results.

[Read the federated diagnosis guide](../experiments/federated_diagnosis/README.md).

## Expanded V2 Stage A/B/C: different dataset

Expanded V2 Stage B/C is NOT the BudgetWise Tier A V2
directory.

The later Chapter 5 evidence reports a separate evaluation
with 30 participants, 254 owner-confirmed transactions
and owner-written Smart Notes.

Its positive genuine-note results are complementary
centralised evidence. They do not establish that the
federated Proposed method outperforms FedAvg or FedProx.

The complete Expanded V2 study evidence must be maintained
with its own reports, source-data provenance and verification
records. This repository index is not a substitute for that
evidence package.

Stage C uses offline pre-collected-note replay, not live
participant prompting.

## PocketIQ deployment

[Deployment code](../deployment/) uses a separately trained
14-category ONNX model. It does not deploy the primary
Tier A federated MLP.

## Evidence and reproducibility rules

1. Preserve all frozen datasets, splits, predictions and results.
2. Keep primary, exploratory and deployment metrics separate.
3. Do not overwrite negative, historical or blocked results.
4. Keep research artifacts and provenance where available.
5. Report missing checkpoints or raw evidence honestly.
6. Do not present retrospective notes as elicited responses.
7. Do not use V2 validation findings to claim held-out gains.

Some intermediate CSVs, checkpoints and other artifacts
are intentionally excluded by the repository .gitignore.
Published summary files are not a complete reproduction
archive.