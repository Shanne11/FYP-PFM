# Reset-Adam versus Persistent Client-Local Adam

## Controlled configuration

Both variants used:

- seed 42;
- the same initial global model (SHA-256 `00a938248d2c12e2d14a7e1a58f3b04f4f6b78d7da061a3b25e7c455ebd8ed3d`);
- the 68 metadata features and no Smart Note features;
- the same 105-hidden-layer architecture adapted only at the input layer (`68 → 128 → 64 → 13`);
- standard cross-entropy;
- Adam learning rate 0.001;
- 150 fixed clients in the same order, with every client participating;
- three local epochs and batch size 32;
- sample-count FedAvg;
- 25 communication rounds; and
- validation-only evaluation.

The only deliberate difference was Adam optimiser-state handling:

- **Reset Adam:** a new Adam optimiser was created for each client in each round.
- **Persistent client-local Adam:** each client retained its own Adam moments and step counter across rounds. Model parameters were reloaded from the current global model before that client's local training. Optimiser states were never shared between clients.

The identical first-round metrics and losses provide an execution-level check that the initial model, batches and first local updates were matched.

## Results

| Variant | Round | Training loss | Validation loss | Accuracy | Macro F1 | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Reset Adam | 1 | 2.5499 | 2.5447 | 0.0925 | 0.0382 | 6 |
| Persistent Adam | 1 | 2.5499 | 2.5447 | 0.0925 | 0.0382 | 6 |
| Reset Adam | 4 | 2.4740 | 2.4628 | 0.2082 | 0.0401 | 2 |
| Persistent Adam | 4 | 2.4760 | 2.4655 | 0.2030 | 0.0392 | 2 |
| Reset Adam | 10 | 2.2735 | 2.2702 | 0.2169 | 0.0274 | 1 |
| Persistent Adam | 10 | 2.2562 | 2.2565 | 0.2169 | 0.0274 | 1 |
| Reset Adam | 25 | 2.1202 | 2.1246 | 0.2163 | 0.0309 | 2 |
| Persistent Adam | 25 | 2.0886 | 2.0978 | 0.2163 | 0.0317 | 2 |

Peak validation results:

| Variant | Best round | Accuracy | Macro F1 | Coverage |
|---|---:|---:|---:|---:|
| Reset Adam | 2 | 0.1463 | 0.0404 | 5 |
| Persistent Adam | 2 | 0.1446 | 0.0398 | 6 |

## Optimiser-state audit

| Round | Reset step range | Persistent step range |
|---:|---:|---:|
| 1 | 3–9 | 3–9 |
| 2 | 3–9 | 6–18 |
| 25 | 3–9 | 75–225 |

The reset variant's step counters restart each round. Persistent step counters accumulate according to each client's own number of local batches. Separate model and optimiser objects were maintained for all clients, and automated tests verify that training one client does not create or change another client's optimiser state.

## Interpretation

Persistent Adam lowers training and validation cross-entropy slightly by round 25, but it does not produce a meaningful classification improvement:

- peak Macro F1 is slightly lower than reset Adam;
- both variants collapse to one predicted class at round 6;
- both have one-class coverage at round 10;
- both finish with identical accuracy and two-class coverage; and
- the round-25 Macro F1 difference is only 0.00083.

Persistent state recovers a second predicted category two rounds earlier, but this does not constitute stable minority-category learning. Replication with seeds 52 and 62 is not justified under the conditional rule.

## Recommended next experiment

The next controlled experiment should target class balance rather than aggregation. Compare standard cross-entropy with the already implemented effective-number loss using:

- reset Adam;
- seed 42 first;
- metadata-only features;
- the same 25-round schedule and client map; and
- validation-only selection.

This directly tests whether the cross-entropy objective is driving majority-class collapse. Aggregation should not be changed yet because client partitions contain 8–13 categories and the earlier matched heuristic-aggregation ablation did not change final predictions.

No held-out test evaluation was performed, and this negative validation result does not resolve the overall FYP accuracy problem.
