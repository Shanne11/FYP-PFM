# Proposed Controlled Training Experiments — Approval Required

No experiment below has been launched.

## Stage A — Centralised control

Keep the current 105→128→64→13 architecture and compare, on validation data:

1. Adam at `0.001` versus SGD with momentum at a small predefined learning-rate grid.
2. 10, 30, and 60 epochs with early stopping by validation Macro F1.
3. standard, inverse-frequency, and effective-number cross-entropy.
4. metadata-only versus all-105 features.

Purpose: establish a learnable reference and identify whether persistent optimiser state or class balancing drives the result.

Estimated CPU time: 5–15 minutes for one seed; 20–45 minutes for three seeds, depending on the grid retained after the first seed.

## Stage B — Matched FedAvg diagnosis

Change one factor at a time from the frozen 10-round/3-local-epoch baseline:

1. rounds: 10 → 25 → 50;
2. local epochs: 1, 3, 5;
3. optimiser: current reset-Adam versus SGD with momentum;
4. learning rate: a small validation-only grid;
5. loss: standard versus effective-number.

All 150 clients, fixed client mapping, splits, feature order, and seeds remain unchanged. The held-out test partition remains untouched until a configuration is frozen.

Estimated CPU time: first run one seed as a 10–25 minute screen. A 25-round configuration across all three seeds is estimated at roughly 30–75 minutes on the present CPU; two matched optimiser configurations are therefore about 1–2.5 hours. Timing should be measured from the one-seed screen before approving anything larger. A full Cartesian sweep is not recommended.

## Stage C — Proposed components

Only after Stage B establishes a stronger FedAvg base:

1. rerun FedProx with the selected base settings;
2. compare no-note, unconditional-note, and RGSF variants on identical ACTM-selected records;
3. add bounded heuristic aggregation last, with a matched no-adjustment ablation.

This order prevents a weak base classifier from being confused with ACTM, Smart Note, or aggregation effects.

## Recommended first experiment

The single most informative next experiment is a matched optimiser/schedule study using standard cross-entropy and metadata-only features: current reset-Adam versus client-local SGD with momentum at 25 communication rounds. It isolates federated optimisation without introducing note quality or class-weight effects.
