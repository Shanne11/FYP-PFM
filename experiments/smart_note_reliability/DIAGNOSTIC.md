# Tier A diagnostic before the experimental extension

Baseline source commit: `c2a71f884d20cc6e52f6bfdacd8be56e07c7947d`.

## Verified pipeline

- `data/experiment_split.json` fixes a category-stratified 7,494/1,729/2,308 train/validation/test split.
- `utils/experiment_data.py` validates a dataset fingerprint, disjoint source-row indices and complete coverage.
- `utils/proposed_features.py` fits category encoding, one-hot metadata encoding, scaling and Smart Note TF-IDF on training rows only. The enforced Tier A contract is 68 metadata plus 37 note features and 13 labels.
- `train_proposed.py` uses 150 fixed `user_id` clients, ten rounds, three local epochs, ACTM thresholds 0.65/0.15 and a ceil-based nominal 30% prompt budget.
- Validation Macro F1 selects the checkpoint; held-out test results are produced afterward.

## Evidence-supported findings

1. The training class ratio is 1,623 Food records versus 184 Savings records (8.82:1); imbalance is material.
2. Canonical mapping produces the expected 13 labels in every split. Transformed matrices are finite and labels stay in class order 0–12. No label-map or train/inference dimensional mismatch was found.
3. Existing Proposed results average accuracy 0.1837, Macro F1 0.0432 and 2.33 actively predicted classes. Inverse-frequency loss raises Macro F1 to 0.0661 and coverage to 6.33 classes but lowers accuracy to 0.1502.
4. A read-only centralized logistic diagnostic on the same frozen features reaches validation Macro F1 0.1182 (metadata) and 0.1318 (all notes), with 11 predicted classes. Balanced logistic regression reaches Macro F1 0.1380–0.1417 and all 13 classes, although accuracy falls to about 0.12. This indicates that the features contain some signal and that the current federated MLP/training schedule collapses more severely than a simple centralized linear diagnostic.
5. Smart Note evidence is weak: training has 6,186 non-empty notes but only 26 unique non-empty values; 1,195 are obvious placeholders (`test`, `!!!`, `...`, `xyz123`, `asdfgh` or `note`). Training note/category mutual information is 0.0167 and weighted note purity is 0.2200, close to the 0.2166 majority share.
6. Exact available-record signatures do not overlap across splits. Transaction IDs overlap by 106/166/40 for train-validation/train-test/validation-test, so IDs are not globally unique and merchant-disjoint independence is not established.
7. BudgetWise Tier A has no merchant-description field. Its lexical anchor is constructed from transaction type, payment mode and location, so a genuine note-to-description semantic relevance feature cannot be evaluated in this dataset.

## Diagnostic conclusion

The low result is consistent with multiple limitations: moderate-to-severe imbalance, federated MLP underfitting/collapse, limited metadata, and weak retrospective note proxies. The audit did not find evidence of an incorrect label mapping, non-finite preprocessing, a feature dimension mismatch or test-fitted TF-IDF. These findings are diagnostic associations, not causal proof.
