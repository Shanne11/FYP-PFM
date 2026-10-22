# Simulated-Client Distribution Analysis

The 7,494 training records are partitioned by the dataset's `user_id` into 150 simulated clients.

| Measure | Result |
|---|---:|
| Records per client | 30–70 |
| Median records per client | 50 |
| Classes per client | 8–13 |
| Median classes per client | 12 |
| Single-class clients | 0 |
| Clients with at most three classes | 0 |
| Mean dominant-class share | 0.2393 |
| Mean normalised label entropy | 0.8417 |

## Interpretation

The client partitions are small, but they are not pathologically label-isolated. Most clients contain nearly all categories, and no client contains only one or a few categories. Therefore, label heterogeneity exists but is not strong enough to explain the prediction collapse by itself.

The more defensible diagnosis is the interaction of:

1. small local sample counts;
2. only three local epochs;
3. independently reset Adam state for each local optimisation;
4. nonlinear parameter averaging;
5. globally weak input features; and
6. moderate 8.82:1 class imbalance.

Because all 150 clients participate in every round, variable participation and client-sampling bias are absent from the current experiment. Sample-count weights range only with the 30–70 record sizes; larger clients contribute more, as ordinary FedAvg specifies.

These clients are simulated research partitions. The distribution must not be presented as evidence of real multi-account user behaviour.

The complete per-client counts are in `outputs/client_distribution.csv`; the compact summary is in `outputs/client_distribution_summary.json`.
