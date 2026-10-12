# Section 16 Evidence Audit

Python repository HEAD: `c421ee22e6af0af7afcae4159626f66ae4a1989e` on `main`.
Working tree clean: **False** (366 changed/untracked paths).
Evidence manifest: **26/26 SHA-256 hashes match**; status remains `provisional-uncommitted-changes`.

## Critical decisions

- Q11 reproduced exactly: 1736 unique cases, 56 wrong→correct, 54 correct→wrong and 315 changed predictions.
- Q12 needs correction: the supplied formula reproduces 874; the stricter implemented metric reproduces 845; neither is 900.
- Q15 needs correction: matched per-client final-test accuracy gives 0 improved, 0 declined and 450 unchanged client-seed cases.
- Q18 passes source inspection and a direct-call test, but remains provisional until the Flutter changes are committed.
- Tier A and PocketIQ are separate model/evidence pathways; the Tier A demonstration trace must not be called a Flutter local audit.

## Status table

| Question | Status | Actual output |
|---|---|---|
| Stage 1 | Verified with limitation | 26/26 hashes match; manifest head=c421ee22e6af0af7afcae4159626f66ae4a1989e; current head=c421ee22e6af0af7afcae4159626f66ae4a1989e; working tree clean=False. |
| Q1 | Verified | Research contract is 105 features/13 classes; deployment contract is 4,096 features/14 classes. |
| Q2 | Verified with limitation | The fixed BudgetWise 'notes' column is read retrospectively; no participant-response collection exists in Tier A. |
| Q3 | Verified with limitation | Outputs are retrospective proxy/selection metrics; no participant usability dataset is present. |
| Q4 | Verified | train=7494, validation=1729, test=2308, training clients=150, labels=13. |
| Q5 | Verified with limitation | Proposed standard: accuracy=0.183709, Macro F1=0.043183, active classes=2.33, zero-recall=10.67. Class-weighted: accuracy=0.150202, Macro F1=0.066090, active classes=6.33, zero-recall=7.67. Train max/min canonical class counts=1623/184 (ratio 8.82). |
| Q6 | Verified with limitation | Repeated transaction-ID values across train–validation=106, train–test=166 and validation–test=40; exact available-record overlap counts=[0, 0, 0]; normalised note-value overlap=[24, 24, 24]. IDs are not globally unique; merchant-disjoint and near-duplicate independence have not been established. |
| Q7 | Verified | Encoders, scaler and note TF-IDF are fitted in ProposedFeatureBuilder.fit(train); validation/test use transform_parts without refitting. |
| Q8 | Verified | ceil(0.30*2308)=693; every strategy/seed selected 693=True. |
| Q9 | Verified | Five strategies present; all select 693; all compare the same 2,308 IDs per seed=True. Three-seed recalculation matches the saved selector summary. |
| Q10 | Verified with limitation | Prompt precision=0.848004; source note used=1736/2079=0.835017. |
| Q11 | Verified with limitation | cases=1736, duplicates=0, wrong→correct=56, correct→wrong=54, changed=315. |
| Q12 | Needs correction | Supplied formula (confidence_after > confidence_before and final incorrect)=874; implemented more_confident_still_wrong (wrong before and after)=845; true-class-probability-increased but final wrong=601. |
| Q13 | Verified | All 30 seed-round comparisons match=True; each method has 150 clients in every round. |
| Q14 | Verified | multiplier range=[0.912590, 0.931737], all within bounds=True; coefficients nonnegative=True; max \|sum-1\|=8.216e-15. |
| Q15 | Needs correction | Reproduced client-seed cases=450: improved=0, declined=0, unchanged=450. Saved global accuracy and Macro F1 are identical per seed. |
| Q16 | Verified with limitation | sample-count/relative-adjustment correlation=0.095067; see quartile table for positive and negative adjustments in each size group. |
| Q17 | Verified with limitation | The implementation compares different model states, so the component combines representation and local-adaptation effects. |
| Q18 | Verified with limitation | remoteSyncEnabled=false; train payload creation=False; pending-row creation=False; sync throws=True; network call=False; two privacy tests passed=True. |
| Q19 | Verified | Research=105 features/13 classes; deployment=4096 features/14 classes; distinct=True. |
| Q20 | Verified with limitation | Class weighting raises proposed mean Macro F1 from 0.043183 to 0.066090 while mean accuracy falls from 0.183709 to 0.150202; full vs no-adjustment has identical accuracy/Macro F1 per seed. |

The CSV evidence matrix contains paths, SHA-256 hashes, checks and limitations for every row.

## Q6 split-overlap correction (12 October 2026)

The original evidence-matrix narrative incorrectly reported zero overlapping transaction-ID values. The frozen `split_overlap_checks.csv` actually reports **106** (train–validation), **166** (train–test), and **40** (validation–test) repeated ID values. These are overlaps of ID *values*, not evidence of identical transactions, because transaction IDs are not globally unique. The available-field exact-record comparison reports **zero** overlaps in each pair; normalised source-note values overlap **24** times in each pair. The dataset has no reliable merchant identifier or separate merchant-description field, so **merchant-disjoint splitting and the absence of semantic near-duplicate leakage are not established**. Keep Q6 as **Verified with limitation**, not an unqualified exclusion of partition leakage.
