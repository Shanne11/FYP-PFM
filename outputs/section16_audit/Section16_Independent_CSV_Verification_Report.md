# Section 16 — Independent Original-CSV Verification

**Date:** 12 October 2026  
**Scope:** Q8–Q12 and Q15, focusing on numerical claims in the supervisor's 20-question analysis.  
**Source:** `Original_Prediction_ACTM_CSVs_Seeds_42_52_62.zip` supplied by the student.

## Integrity and coverage

- Outer ZIP SHA-256: `bdfd89d3976e505938b8533d713654c636505107f0b526fecc01e49f5c6874b3`, matching the separately uploaded checksum file.
- All **28 of 28** entries in `MANIFEST.json` match their recorded SHA-256 hashes, byte sizes, and CSV row counts where provided.
- The ACTM transaction-outcome file has **34,620 rows**: 2,308 held-out transactions × 5 strategies × 3 seeds.
- Each seed/strategy group selects exactly **693** records. Each has 2,308 distinct `source_index` values and no duplicate `(seed, strategy, source_index)` key.
- All `note_used` rows are both prompted and note-available.

## Q9 — Five selection strategies

The following three-seed means were recalculated from transaction-level rows. They agree with the saved ACTM run-level files (to floating-point precision).

| Strategy | Prompt precision | Wrong-to-correct per 100 prompts | Mean net corrections per seed | Overall accuracy |
|---|---:|---:|---:|---:|
| Random | 0.813853 | 1.250601 | −0.333333 | 0.183276 |
| Entropy-only | 0.860991 | 0.721501 | −1.000000 | 0.182987 |
| Margin-only | 0.828283 | 3.126503 | −1.000000 | 0.182987 |
| Conflict-only | 0.817220 | 1.250601 | +0.333333 | 0.183564 |
| Proposed ACTM | 0.848004 | 2.693603 | +0.666667 | 0.183709 |

These results support **trade-offs**, not universal superiority. The saved run results show Conflict-only has the highest mean Macro F1 (approximately 0.044025); the raw table's predicted/actual class labels could be used to reproduce this with a compatible Macro F1 implementation.

## Q10 — Proxy-note selection and uncertainty

For Proposed ACTM, there are **2,079 prompted transaction-seed cases** and **1,736 note-used cases**, giving note acceptance **83.5017%**. Of the 2,079 prompted cases, **1,763 were initially incorrect**, giving proxy-defined prompt precision **84.8004%**. The three original per-seed `prompt_metrics.csv` files report an average mean uncertainty reduction of **0.000275539**, rounding to **0.000276**. The uncertainty values are saved run-level results; this package lacks full probability vectors needed to independently recompute entropy-based uncertainty change from scratch.

These quantities do **not** measure human note acceptance or establish that a note was genuinely useful.

## Q11 — Before/after classification changes

The Proposed ACTM `note_used == True` rows total **1,736 transaction-seed cases**. Counts independently reproduced from `correct_before`, `correct_after`, and before/after predicted labels:

| Outcome | Cases |
|---|---:|
| Wrong → Correct | 56 |
| Correct → Wrong | 54 |
| Wrong → Different Wrong | 205 |
| Wrong → Same Wrong | 1,214 |
| Correct → Same Correct | 207 |
| **Total** | **1,736** |

There are **315 category changes** (56+54+205, or 18.145% of note-used cases), and a net gain of just **2 correctly categorised transaction-seed cases**. These are retrospective source-note-proxy outcomes, not participant results.

## Q12 — Confidence increased but final prediction wrong

For the same 1,736 note-used Proposed ACTM transaction-seed cases:

- **874 (50.35%)** satisfy `confidence_after > confidence_before` **and** `correct_after == False` (supervisor-aligned definition).
- **845 (48.68%)** satisfy the above **and** `correct_before == False` (strictly wrong both before and after).
- The earlier **900** figure is not supported by the original CSV rows and was correctly removed from Section 16.

## Q15 — Full versus no-heuristic-adjustment paired ablation

Each seed's paired `full/predictions.csv` and `without_heuristic_adjustment/predictions.csv` contains 2,308 identically ordered transaction IDs and exactly matching actual labels.

| Seed | Predicted-category differences | Final accuracy (both arms) | Macro F1 (both arms) |
|---|---:|---:|---:|
| 42 | 0 | 0.199740 | 0.043342 |
| 52 | 0 | 0.170277 | 0.037328 |
| 62 | 0 | 0.181109 | 0.048880 |

The prediction categories are **identical row by row**, although most paired confidence values differ slightly. Therefore, no client's final-test accuracy can differ **provided the matching client-to-test-record mapping is shared**, as stated by the prior client-level audit. Its **0 improved, 0 declined, 450 unchanged** counts are consistent with the original predictions, but this ZIP alone does not contain the per-client mapping necessary to reconstruct those 450 comparisons independently.

## Limitations and next steps

1. No new experiments need to be run to verify the listed classification counts.
2. The frozen source commit IDs, source code, and matched client assignment data are outside this particular ZIP; code-level and per-client provenance still rely on the separate provisional evidence package and eventual commits.
3. The numerical verification does not establish user-entered Smart Note effectiveness, genuine ambiguity-detection validity, merchant-disjoint splitting, privacy guarantees, or statistical superiority.
4. Section 16 Q12 and Q15 in the connected Google Doc were already corrected, and the CSV counts agree with those corrections.
5. Finalise both Git repositories, update the final SHA-256 evidence manifest, and keep this report alongside the original CSV evidence.

**Reproducible check:** Run `python Section16_Independent_Verification.py Original_Prediction_ACTM_CSVs_Seeds_42_52_62.zip` with pandas installed. The companion console output contains the full check summary.
