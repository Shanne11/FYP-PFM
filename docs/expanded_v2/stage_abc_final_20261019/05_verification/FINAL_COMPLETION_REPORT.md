# Expanded V2 final evidence-closure report

## Outcome

The closure was completed from frozen artifacts only. No training, tuning, label editing, record removal or prediction modification occurred.

| model | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | weighted_f1_mean | weighted_f1_std |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.6322 | 0.0701 | 0.2949 | 0.1183 | 0.6299 | 0.0698 |
| B1 | 0.6463 | 0.0757 | 0.2966 | 0.1156 | 0.6430 | 0.0712 |
| B2 | 0.7910 | 0.0624 | 0.3862 | 0.1290 | 0.7860 | 0.0575 |

Across 214 transaction-seed cases, B1→B2 yielded 32 Wrong-to-Correct, 1 Correct-to-Wrong, 137 Correct-to-Correct and 44 Wrong-to-Wrong transitions, for 31 net corrections.

**Genuine owner-written Smart Notes improved frozen-model transaction classification in the Expanded V2 participant-disjoint evaluation.**

Stage C is offline pre-collected-note replay. Random, Entropy and Margin results are reported for every nominal budget and G0/G1/G2. Learned Benefit and G3 remain blocked/not evaluated.

## Evidence limitations

- The shareable archive excludes restricted raw Forms data, consent/withdrawal registries, transaction descriptions, Smart Note text and fitted model binaries.
- Full training reproduction therefore requires authorised access to private inputs; saved-prediction result verification is complete.
- Participant-disjoint evaluation is not description-template-disjoint.
- The participant-description graph has one connected component, so component-disjoint sensitivity is infeasible.
- Formal category definitions and exclusion examples were not present in the frozen taxonomy artifacts. Exact labels, label order and documented source mappings are preserved without invented boundaries.
- Housing and Insurance have zero support.
- Written institutional/ethics confirmation is not in the shareable evidence.
