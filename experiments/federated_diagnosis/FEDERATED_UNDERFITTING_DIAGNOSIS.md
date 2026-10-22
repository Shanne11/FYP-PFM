# Phase 2 Federated Underfitting Diagnosis

## Controlled validation comparison

New diagnostics were selected and measured on validation data only.

| Method | Features | Accuracy | Macro F1 | Predicted classes |
|---|---|---:|---:|---:|
| Majority class | None | 0.2169 | 0.0274 | 1 |
| Central logistic | 68 metadata | 0.2296 | 0.1182 | 11 |
| Central logistic, balanced | 68 metadata | 0.1209 | 0.1380 | 13 |
| Central MLP, 3-seed mean | 68 metadata | 0.2383 | 0.1287 | 10.67 |
| Central logistic | All 105 | 0.2331 | 0.1318 | 11 |
| Central logistic, balanced | All 105 | 0.1197 | 0.1417 | 13 |
| Central MLP, 3-seed mean | All 105 | 0.2115 | 0.1368 | 11.67 |
| Historical FedAvg selected checkpoint, 3-seed mean | All 105 | 0.1841 | 0.0448 | Not recorded per checkpoint |
| Historical FedProx selected checkpoint, 3-seed mean | All 105 | 0.1847 | 0.0453 | Not recorded per checkpoint |
| Historical proposed selected checkpoint, 3-seed mean | Selective notes | 0.1895 | 0.0450 | Not recorded per checkpoint |

The centralised MLP uses the same architecture and training-fitted preprocessing as the federated model. Its validation Macro F1 is roughly three times the federated result. This is direct evidence of a federated optimisation gap. It also shows that the architecture can learn more than the federated checkpoints under the present features, so insufficient model capacity is not the primary explanation.

## Root-cause findings

### 1. Federated underfitting and collapse are present

The seed-42 FedAvg validation Macro F1 peaks at 0.0446 in round 4, then falls to 0.0274 by round 10 while accuracy rises to 0.2169—the exact validation majority-class accuracy. The proposed trace behaves similarly. This is a strong signature of convergence toward the dominant class rather than healthy generalisation.

### 2. Class imbalance contributes but is not the only cause

Training counts range from 184 (`Savings`) to 1,623 (`Food`), a ratio of 8.82:1. Balanced logistic regression predicts all 13 categories and improves Macro F1, but accuracy decreases sharply. Earlier inverse-frequency and effective-number experiments show the same trade-off. Therefore, imbalance explains part of the minority-class failure but not the centralised-versus-federated gap.

### 3. Local client distributions are not the primary cause

Every client contains at least eight categories, median 12, and mean normalised label entropy is 0.8417. Non-IID label composition may add noise, but the partitions are substantially more diverse than the extreme single-label-client setting usually associated with federated collapse.

### 4. The feature contract is weak for semantic categorisation

The 68 metadata features omit merchant and transaction description. Most predictive content must come from transaction type, payment mode, location, amount, and date. Central models outperform the federated models but remain weak, demonstrating a dataset/feature ceiling in addition to the federated optimisation problem.

### 5. Existing Smart Notes do not solve the feature weakness

Adding the 37-note block raises central Macro F1 from 0.1287 to 0.1368, but lowers accuracy from 0.2383 to 0.2115. The earlier dataset audit found only 26 unique non-empty note values, 1,195 obvious training placeholders, and note/category mutual information near 0.0167. These retrospective proxies are not sufficiently informative to justify unconditional fusion.

### 6. No silent label, dimension, softmax, split, or aggregation defect was found

The category order, feature dimensions, loss/logit relationship, train-only fitting, and sample-count averaging are consistent. The current results are not explained by an obvious class-index mismatch or validation/test leakage. The main implementation risk is optimisation design, not a silent tensor-contract error.

## Current conclusion

The classifier is underfitting in the federated setting. The most likely causes are short local training on small clients, reset local Adam states followed by nonlinear parameter averaging, moderate class imbalance, and limited input information. Class imbalance is a contributor, not the sole root cause. The BudgetWise note field is also too weak to evaluate Smart Note usefulness convincingly.

## Evidence files

- `outputs/centralised_validation_comparison.csv`
- `outputs/centralised_epoch_metrics.csv`
- `outputs/historical_federated_validation_checkpoints.csv`
- `outputs/historical_federated_heldout_summary.csv`
- `outputs/training_class_distribution.csv`
- `outputs/client_distribution.csv`
