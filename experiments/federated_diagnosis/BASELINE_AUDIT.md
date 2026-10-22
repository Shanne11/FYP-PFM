# Phase 1 Baseline and Repository Audit

## Repository state

- Diagnostic branch: `codex/federated-underfitting-diagnosis`
- Baseline source commit: `8934dfa3e50e3593ee1c0d584e7a4d75230cd5c9`
- Parent chain: `e4aabb5` → `c2a71f8` → `6d17323` → `8934dfa`
- `e4aabb5` is an ancestor of the current branch, so the verified Q6 correction is included.
- The prior experimental branch was two commits ahead of `origin/main` when this branch was created.
- Pre-existing untracked paths were preserved. No merge, reset, rebase, cherry-pick, push, or PocketIQ change was performed.

## Frozen research contract

| Item | Verified value |
|---|---:|
| Training records | 7,494 |
| Validation records | 1,729 |
| Held-out records | 2,308 |
| Simulated training clients | 150 |
| Categories | 13 |
| Metadata features | 68 |
| Smart Note TF-IDF features | 37 |
| Total input features | 105 |
| Communication rounds | 10 |
| Local epochs per round | 3 |
| Learning rate | 0.001 |
| Local batch size | 32 |

The dataset SHA-256 is `608bc42c26f13e1323bc2be4bf614c28968e62a1aa873911d243c462fb4b740d`; the frozen split manifest SHA-256 is `861960e6f787f60870610dd564dc20c741d4dadda265d46dc57e4dce7f164cd0`.

## Implementation correctness audit

- The MLP is `105 → 128 → 64 → 13`, with ReLU and dropout 0.3. It returns logits, which is correct for PyTorch cross-entropy.
- One label encoder fitted on training categories is reused for validation and held-out transformations. All partitions use the same 13-class order.
- The categorical encoder, numeric scaler, and Smart Note TF-IDF vocabulary are fitted on the training partition only.
- Metadata consists of transaction type, payment mode, location, amount, and calendar-derived fields. The Tier A dataset supplies no merchant or transaction-description feature to this model.
- Numeric values are standardised; categoricals are one-hot encoded; notes use TF-IDF. The resulting dimensions match the frozen 68 + 37 contract.
- FedAvg uses non-negative sample-count weights that sum to one. There are no batch-normalisation buffers requiring special aggregation.
- FedProx adds the proximal penalty against the current global parameter state. Its matched performance is almost identical to FedAvg, so the configured `mu=0.01` does not resolve the collapse.
- All 150 clients participate in every round. Client sampling is therefore not a cause in the current runs.
- Global training class weights, when enabled, are calculated from the training labels only.
- Checkpoint selection uses validation Macro F1. Newly added Phase 2 centralised diagnostics did not evaluate the held-out test partition.

## Important optimisation observation

Each client receives a copy of the global model and a newly created Adam optimiser in each round. The optimiser state is therefore reset 1,500 times per seed (150 clients × 10 rounds). Each local dataset has only 30–70 records and is trained for three epochs before parameters are averaged. This implementation is internally consistent, but it is a difficult optimisation regime for Adam and a nonlinear MLP. It is a plausible contributor to underfitting and requires controlled validation experiments rather than being labelled a coding error.

## Comparison-validity boundaries

- Centralised and federated diagnostics use the same frozen split, category order, feature builder, architecture, learning rate, and seeds.
- The centralised MLP receives 30 complete training epochs, matching the federated schedule's nominal 10 × 3 local passes. The number and order of optimiser updates are not identical; this is a diagnostic comparison, not a claim of exact computational equivalence.
- Historical federated held-out results are copied, not regenerated. New method selection must remain validation-only until a configuration is frozen.

Machine-readable provenance is in `outputs/diagnostic_contract.json`.
