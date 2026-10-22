# Stage 4: Three-Seed Effective-Number Replication

## Scope and controls

This validation-only experiment compares standard cross-entropy (CE) with effective-number class-balanced cross-entropy at seeds 42, 52 and 62. Both configurations use the fixed BudgetWise training/validation split, the existing 150-client map, 68 metadata features, the 68-128-64-13 MLP, reset Adam at learning rate 0.001, three local epochs, batch size 32, all-client participation, sample-count FedAvg and 25 communication rounds. The effective-number configuration uses beta = 0.9999 and training-partition class counts, with weights normalised to arithmetic mean 1.

Seed 42 is reused from Stage 3 only after the script verifies all 17 Stage 3 artifact hashes and checks the relevant configuration. Seeds 52 and 62 were executed in this stage. The held-out test partition was neither transformed nor evaluated.

## Round-25 results

| Seed | Loss | Accuracy | Macro F1 | Weighted F1 | Predicted categories | Zero-recall categories |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Standard CE | 0.2163 | 0.0309 | 0.0867 | 2 | 11 |
| 42 | Effective number | 0.2296 | 0.0947 | 0.1457 | 6 | 7 |
| 52 | Standard CE | 0.2256 | 0.0451 | 0.0942 | 4 | 11 |
| 52 | Effective number | 0.2400 | 0.0725 | 0.1322 | 6 | 10 |
| 62 | Standard CE | 0.2192 | 0.0372 | 0.0895 | 3 | 11 |
| 62 | Effective number | 0.2337 | 0.0829 | 0.1377 | 8 | 6 |

The paired effective-number minus standard-CE differences were positive for every seed:

| Seed | Accuracy difference | Macro-F1 difference | Weighted-F1 difference | Coverage difference |
|---:|---:|---:|---:|---:|
| 42 | +0.0133 | +0.0638 | +0.0590 | +4 |
| 52 | +0.0145 | +0.0274 | +0.0380 | +2 |
| 62 | +0.0145 | +0.0457 | +0.0483 | +5 |

## Three-seed aggregate

Values are sample mean plus/minus sample standard deviation across the three seeds.

| Loss | Accuracy | Macro F1 | Weighted F1 | Predicted-category coverage | Zero-recall categories |
|---|---:|---:|---:|---:|---:|
| Standard CE | 0.2204 +/- 0.0047 | 0.0377 +/- 0.0071 | 0.0901 +/- 0.0038 | 3.00 +/- 1.00 | 11.00 +/- 0.00 |
| Effective number | 0.2344 +/- 0.0052 | 0.0834 +/- 0.0111 | 0.1386 +/- 0.0068 | 6.67 +/- 1.15 | 7.67 +/- 2.08 |

Effective-number loss therefore increased the three-seed mean round-25 accuracy by 0.0141, Macro F1 by 0.0457 and Weighted F1 by 0.0484, while increasing mean coverage by 3.67 categories. These are descriptive results from three seeds and are not evidence of statistical significance.

## Stability and peak performance

| Seed | Standard peak Macro F1 (round) | Effective-number peak Macro F1 (round) | Effective late-round minimum coverage (rounds 16-25) |
|---:|---:|---:|---:|
| 42 | 0.0404 (2) | 0.0978 (21) | 5 |
| 52 | 0.0524 (5) | 0.1048 (19) | 6 |
| 62 | 0.0422 (2) | 0.1160 (17) | 6 |

The standard objective entered single-category collapse in seeds 42 and 62; seed 52 avoided a literal one-category state but remained severely restricted. Effective-number loss did not enter a single-category state in any of the three seeds and maintained at least five or six predicted categories during rounds 16-25. Its benefit was therefore sustained to the fixed round-25 budget, although seed 52 declined from its peak Macro F1 of 0.1048 to 0.0725 at round 25.

## Per-class interpretation

Standard CE assigned non-zero recall to only two validation categories in every seed at round 25, although small extra prediction coverage did not always translate into non-zero recall. Effective-number training improved recall for several categories, particularly Rent in all seeds and seed-dependent gains for Bonus, Investment and Freelance. The gains were not uniform: Entertainment, Health, Savings and Utilities had zero recall in all effective-number runs, and seed 52 still had ten zero-recall categories. This is reduced collapse, not complete category recognition.

## Comparison with the earlier centralised MLP

The earlier centralised metadata-only MLP validation benchmark reported approximately 0.2383 accuracy and 0.1287 Macro F1. The effective-number federated mean accuracy of 0.2344 approaches that accuracy, but its mean Macro F1 of 0.0834 remains materially lower. This comparison is diagnostic rather than a fully matched causal comparison because the centralised and federated optimisation processes differ. It indicates that federated objective balancing reduces collapse but does not remove the remaining federated learning gap.

## Decisions

1. **Consistency:** Effective-number loss improved round-25 accuracy, Macro F1, Weighted F1 and coverage in every seed.
2. **Unrecognised categories:** Six to ten categories per seed retained zero recall; the effective-number mean was 7.67 of 13.
3. **Collapse:** Literal single-class collapse was eliminated in the three effective-number validation runs, but substantial multi-class collapse remains.
4. **Smart Note readiness:** The classifier is a more stable foundation than standard CE, but weak and seed-sensitive minority recognition remains. These validation results alone do not justify a new Smart Note performance claim.
5. **Next research step:** Because prior analysis found the BudgetWise source notes sparse and weakly informative, the scientifically stronger next step is to investigate better transaction and clarification data before conducting a new Smart Note experiment. BudgetWise can remain the historical classifier benchmark.

No Smart Note fusion, ACTM, RGSF, utility-weighted aggregation, replacement dataset or held-out test experiment was run in this stage.
