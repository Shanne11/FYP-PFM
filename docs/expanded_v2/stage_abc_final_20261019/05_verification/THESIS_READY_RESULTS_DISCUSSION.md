# Thesis-ready results, discussion and limitations

## Stage B results

Across five participant-disjoint seeded evaluations, B1 achieved mean accuracy 0.6463 (SD 0.0757) and mean Macro-F1 0.2966 (SD 0.1156). The same frozen unified model with owner-written Smart Notes (B2) achieved mean accuracy 0.7910 (SD 0.0624) and mean Macro-F1 0.3862 (SD 0.1290). Across 214 transaction-seed cases, the comparison produced 32 Wrong-to-Correct, 1 Correct-to-Wrong, 137 Correct-to-Correct and 44 Wrong-to-Wrong outcomes, for 31 net corrections.

**Conclusion preserved:** Genuine owner-written Smart Notes improved frozen-model transaction classification in the Expanded V2 participant-disjoint evaluation.

This statement is bounded to this dataset, model and evaluation. It does not establish universal effectiveness or causal generalisation. Per-seed participant-clustered intervals are reported separately; several Macro-F1 intervals include zero, and the five seeded partitions overlap.

## Stage C results

Stage C is **OFFLINE PRE-COLLECTED-NOTE REPLAY**. Random, Entropy and Margin were compared at nominal 5%, 10%, 20% and 30% budgets using the same held-out records and realised counts. G0 retains B1, G1 always uses an available selected note, and G2 applies the validation-tuned rule. Margin obtained the largest descriptive net correction at 30% under G1, but the study does not establish statistical superiority of Margin and did not measure live prompt acceptance, response burden or time savings. Learned Benefit and G3 were blocked.

## Category support

Twelve of fourteen frozen categories were observed. Housing and Insurance had zero records and are not estimable. Several categories were sparse; consequently, Macro-F1 is sensitive to rare/absent classes and remains modest even as overall accuracy improves.

## Template sensitivity

Participant separation was preserved, but description templates overlap across partitions. The closure quantifies overlap and reports a frozen-prediction overlapping/non-overlapping sensitivity where records are available. Because the participant-description graph has one connected component, a component-disjoint re-evaluation is infeasible without new independent data.

## Limitations

- only 30 participants and 254 transactions;
- repeated participants and transactions across seeded evaluations prevent treating 214 transaction-seed cases as independent;
- owner-confirmed labels were not externally adjudicated;
- no per-note response times;
- all collected notes were answered, so skip/non-response behaviour is not represented;
- template overlap remains;
- two categories have no observations;
- Stage C is offline replay rather than live clarification;
- no claim of universal Smart Note effectiveness, Margin superiority, live prompting success or user-time savings.

## Main numerical tables

### Stage B

| model | accuracy_mean | accuracy_sd | macro_f1_mean | macro_f1_sd | weighted_f1_mean | weighted_f1_sd | ece_mean | brier_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.6322 | 0.0701 | 0.2949 | 0.1183 | 0.6299 | 0.0698 | 0.2155 | 0.5380 |
| B1 | 0.6463 | 0.0757 | 0.2966 | 0.1156 | 0.6430 | 0.0712 | 0.1679 | 0.5023 |
| B2 | 0.7910 | 0.0624 | 0.3862 | 0.1290 | 0.7860 | 0.0575 | 0.1225 | 0.3407 |

### Per-seed transitions

| seed | Wrong->Correct | Correct->Wrong | Correct->Correct | Wrong->Wrong | net_corrections | evaluated_records |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 6 | 1 | 30 | 6 | 5 | 43 |
| 52 | 6 | 0 | 29 | 6 | 6 | 41 |
| 62 | 7 | 0 | 26 | 11 | 7 | 44 |
| 72 | 7 | 0 | 24 | 13 | 7 | 44 |
| 82 | 6 | 0 | 28 | 8 | 6 | 42 |
