# Stage 5: Fixed-Configuration Held-Out Evaluation

## Scope and evidence boundary

This exploratory extension evaluates the frozen Stage 4 standard-CE and effective-number configurations on the existing BudgetWise held-out test partition. The partition had already been inspected in earlier Tier A work, so this is not a pristine independent confirmatory test. No held-out result was used to tune beta, architecture, learning rate, local epochs, communication rounds or model-selection rules.

Before held-out scoring, all 150 validation-round records from the six deterministic training replays matched the Stage 4 trajectories with maximum absolute difference no greater than 1e-12. Held-out features and labels were transformed only after all six fixed models completed and passed that replay check.

## Fixed round-25 held-out results

| Seed | Loss | Accuracy | Macro F1 | Weighted F1 | Coverage | Zero-recall count | Zero-recall categories |
|---:|---|---:|---:|---:|---:|---:|---|
| 42 | Standard CE | 0.2171 | 0.0308 | 0.0864 | 2 | 11 | Bonus;Education;Entertainment;Freelance;Health;Investment;Other;Salary;Savings;Travel;Utilities |
| 42 | Effective number | 0.2253 | 0.0912 | 0.1387 | 5 | 8 | Education;Entertainment;Freelance;Health;Other;Salary;Savings;Utilities |
| 52 | Standard CE | 0.2262 | 0.0459 | 0.0943 | 4 | 11 | Bonus;Education;Entertainment;Freelance;Health;Investment;Rent;Salary;Savings;Travel;Utilities |
| 52 | Effective number | 0.2392 | 0.0739 | 0.1319 | 6 | 8 | Bonus;Education;Entertainment;Freelance;Health;Salary;Savings;Utilities |
| 62 | Standard CE | 0.2175 | 0.0327 | 0.0864 | 4 | 11 | Bonus;Education;Entertainment;Freelance;Health;Investment;Rent;Salary;Savings;Travel;Utilities |
| 62 | Effective number | 0.2335 | 0.0842 | 0.1413 | 8 | 7 | Education;Entertainment;Health;Investment;Other;Savings;Utilities |

Three-seed means and sample standard deviations:

| Loss | Accuracy | Macro F1 | Weighted F1 | Coverage | Zero-recall categories |
|---|---:|---:|---:|---:|---:|
| Standard CE | 0.2202 ± 0.0051 | 0.0365 ± 0.0082 | 0.0890 ± 0.0046 | 3.33 ± 1.15 | 11.00 ± 0.00 |
| Effective number | 0.2327 ± 0.0070 | 0.0831 ± 0.0087 | 0.1373 ± 0.0048 | 6.33 ± 1.53 | 7.67 ± 0.58 |

Mean paired effective-number minus standard differences were +0.0124 accuracy, +0.0466 Macro F1, +0.0483 Weighted F1 and +3.00 categories of coverage.

## Answers to the Stage 5 questions

1. **Accuracy in every seed:** Yes. Effective-number accuracy was higher in every paired seed.
2. **Macro F1 in every seed:** Yes. Effective-number Macro F1 was higher in every paired seed.
3. **Consistency with validation:** The direction is consistent for both accuracy and Macro F1 across all seeds. Magnitudes and category-specific results remain partition-dependent.
4. **Unrecognised categories:** Effective-number retained zero recall for 7–8 of 13 categories across seeds; see `PER_CLASS_METRICS.csv` for names and values.
5. **Centralised comparison:** The earlier centralised metadata-only MLP benchmark was validation-only (accuracy about 0.2383, Macro F1 about 0.1287). The effective-number held-out mean differs by -0.0056 accuracy and -0.0456 Macro F1, but this is not a matched same-partition comparison and cannot establish that the federated model has reached the centralised baseline.
6. **Remaining classification problem:** Class balancing should be interpreted as reducing collapse only if the per-seed coverage and Macro F1 support that conclusion. The absolute Macro F1 and zero-recall counts show that further classifier/data diagnosis remains necessary; class imbalance is not the only demonstrated limitation.
7. **Smart Note readiness:** These results can identify a stronger base-classifier candidate, but they do not by themselves make BudgetWise suitable for a new Smart Note claim. Earlier evidence found the available note proxies weak and repetitive. A genuine Smart Note study still requires more informative clarification data and a separate protocol.

## Limitations and stopping decision

- Three seeds describe variability but do not prove statistical significance.
- The held-out partition was previously inspected, making this an exploratory extension.
- The comparison isolates the loss function under the frozen federated configuration; it does not evaluate ACTM, Smart Note fusion, RGSF, utility weighting, local learning or PocketIQ.
- No further tuning or Smart Note experiment was started after observing these results.
