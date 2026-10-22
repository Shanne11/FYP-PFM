# Stage 1 Federated Optimisation Screen

## Experimental contract

- Seed: 42
- Features: 68 metadata features; Smart Note features excluded
- Loss: standard cross-entropy
- Model: 68 → 128 → 64 → 13 MLP with ReLU and dropout 0.3
- Clients: fixed 150-client training map; all clients participate
- Local schedule: three epochs, batch size 32
- Full runs: 25 communication rounds
- Selection and evaluation: validation partition only
- Held-out test evaluation: not performed

Both variants recreate the local optimiser for every client in every round. Consequently, this experiment compares optimiser algorithms under the existing reset behaviour; it does not isolate the effect of resetting Adam.

## SGD selection

Momentum was fixed at 0.9. Learning rates 0.01, 0.03 and 0.1 were screened for 10 rounds. The predeclared selection rule was the highest best-round validation Macro F1, with validation accuracy as a tie-breaker.

| SGD learning rate | Best round | Best accuracy | Best Macro F1 | Coverage |
|---:|---:|---:|---:|---:|
| 0.01 | 3 | 0.1336 | 0.0531 | 6 |
| 0.03 | 1 | 0.1319 | 0.0524 | 6 |
| 0.10 | 9 | 0.2175 | 0.0284 | 2 |

The selected configuration was SGD learning rate 0.01 with momentum 0.9.

## Matched 25-round result

| Optimiser | Round | Training loss | Validation loss | Accuracy | Macro F1 | Coverage | Single-class collapse |
|---|---:|---:|---:|---:|---:|---:|---|
| Reset Adam, LR 0.001 | 4 | 2.4740 | 2.4628 | 0.2082 | 0.0401 | 2 | No |
| Reset Adam, LR 0.001 | 10 | 2.2735 | 2.2702 | 0.2169 | 0.0274 | 1 | Yes |
| Reset Adam, LR 0.001 | 25 | 2.1202 | 2.1246 | 0.2163 | 0.0309 | 2 | No, but near-collapse |
| Reset SGD, LR 0.01, momentum 0.9 | 4 | 2.5276 | 2.5202 | 0.1706 | 0.0527 | 6 | No |
| Reset SGD, LR 0.01, momentum 0.9 | 10 | 2.4651 | 2.4589 | 0.2169 | 0.0275 | 2 | No, but near-collapse |
| Reset SGD, LR 0.01, momentum 0.9 | 25 | 2.3445 | 2.3391 | 0.2169 | 0.0274 | 1 | Yes |

Best checkpoint observations:

- Adam's best validation Macro F1 was 0.0404 at round 2, with accuracy 0.1463 and five predicted categories.
- SGD's best validation Macro F1 was 0.0531 at round 3, with accuracy 0.1336 and six predicted categories.
- SGD therefore produced a better early Macro F1 and broader early coverage, but lower accuracy.
- The improvement did not persist. At round 25, SGD predicted only one category and its Macro F1 was below Adam's.

## Interpretation

Neither optimiser gives a durable solution. The cross-entropy loss continues to decrease while category coverage collapses, showing that lower loss is being obtained mainly by fitting the dominant-category distribution rather than maintaining minority-category discrimination.

SGD is not sufficiently promising for seeds 52 and 62 under the approved conditional rule because:

1. its advantage exists only at an early checkpoint;
2. it does not improve accuracy at that checkpoint;
3. it collapses to one category by round 25; and
4. its round-25 Macro F1 is worse than Adam.

Stage 2 replication was therefore not run.

## Recommended follow-up

A reset-Adam versus persistent-state Adam experiment is necessary if the thesis wants to investigate whether optimiser-state resetting contributes to collapse. The present Adam-versus-SGD comparison cannot answer that causal question because it changes the optimiser algorithm as well as the moment/state dynamics.

The follow-up should first use seed 42 and validation data only, with identical Adam hyperparameters and schedules. Per-client Adam moment states would be retained across rounds while client model parameters are reloaded from the current global model. This is a nonstandard design choice that must be described explicitly. It should not be launched without separate approval.

The overall FYP accuracy problem is not solved by this experiment.
