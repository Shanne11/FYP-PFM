# Findings: class-balanced learning and RGSF

## Classifier comparison

The strictly matched loss comparison holds the split, 150-client map, architecture, ACTM, note fusion, aggregation, rounds, local epochs and seeds fixed.

| Loss | Accuracy, mean | Macro F1, mean | Weighted F1, mean |
|---|---:|---:|---:|
| Standard cross-entropy | 0.1837 | 0.0432 | 0.0951 |
| Inverse-frequency cross-entropy | 0.1502 | 0.0661 | 0.0872 |
| Effective-number loss | 0.1570 | 0.0695 | 0.1025 |

Effective-number loss produced the highest mean validation Macro F1 (0.0690) and held-out Macro F1 (0.0695). Relative to standard cross-entropy, it increased Macro F1 by about 0.0263 but reduced accuracy by about 0.0267. Relative to inverse-frequency weighting, it increased Macro F1 by about 0.0034 and accuracy by about 0.0068. With only three seeds, these differences are descriptive and are not claimed as statistically significant.

## Matched Smart Note ablation

| Variant | Accuracy, mean | Macro F1, mean | Wrong→correct, total | Correct→wrong, total | Confident-but-wrong, total |
|---|---:|---:|---:|---:|---:|
| A2 effective-number, no notes | 0.1631 | 0.0715 | 0 | 0 | 0 |
| A3 effective-number, accept every selected note | 0.1567 | 0.0694 | 62 | 72 | 843 |
| A4 effective-number with RGSF | 0.1577 | 0.0702 | 2 | 5 | 29 |

RGSF accepted 31 of 1,705 selected, available source-note cases (1.82%). It substantially reduced harmful note-driven changes and confident-but-wrong outcomes compared with unconditional fusion, but it did not produce a positive net correction count: two wrong-to-correct and five correct-to-wrong changes give a net change of −3 across the three seeds. A4 also remained below the no-note A2 model in accuracy and Macro F1.

## Interpretation

The classifier extension is a trade-off improvement: effective-number loss improves equal-class performance and prediction coverage but does not recover the original accuracy. RGSF is successful as a conservative harm-reduction mechanism relative to unconditional fusion, but unsuccessful as evidence that these retrospective notes improve categorisation. The strongest extension result is the effective-number model without note fusion.

The likely limitation is the note proxy rather than the gate alone: only 26 unique non-empty note strings occur in training, many are placeholders, and their category association is weak. A genuine Smart Note usefulness claim requires a separate licensed dataset with informative clarifications or a real-user clarification study.
