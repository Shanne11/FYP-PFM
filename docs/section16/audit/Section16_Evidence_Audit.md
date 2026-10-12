# Section 16 Evidence Audit — FINAL v2

Documentation reconciliation date: **12 October 2026**  
Scope: **Documentation only. No experiments were rerun, no models were retrained, no verified numerical results were changed and no Flutter implementation was modified.**

## Repository provenance

- Tier A Python research-source commit: `e4aabb55a7d779e2e09dcac59c2a5746895c2149` on `main`; `origin/main` matches; ahead 0, behind 0.
- PocketIQ Flutter research-source commit: `11daf49f491e70f9f7057cc3cf3bc3616990bb96` on `main`; `origin/main` matches; ahead 0, behind 0.
- No source-repository documentation commit was created. This report belongs to the standalone versioned evidence package.
- Python Git status reported no tracked or staged changes and no other reportable untracked paths. Git could not inspect the local `.pytest_cache` directory because of its local access state; that cache is not source or evidence.
- Flutter Git status reported no tracked, staged or untracked changes.

## Reconciled critical findings

- **Q6 — Verified with limitation:** repeated transaction-ID values are 106 (train–validation), 166 (train–test) and 40 (validation–test). Exact available-record overlaps are 0/0/0. Transaction IDs are not globally unique, and merchant-disjoint or semantic near-duplicate independence is not established.
- **Q12 — Verified with limitation:** among 1,736 retrospective note-used transaction-seed evaluations, 874 (50.35%) have increased confidence with a wrong final prediction. Under the stricter wrong-before-and-after definition, 845 (48.68%) qualify. These are source-note proxy outcomes, not real-user clarification outcomes.
- **Q15 — Verified with limitation:** the separate matched client-level audit supports 0 improved, 0 declined and 450 unchanged client-seed cases. Final predicted categories are identical across matched held-out records for seeds 42, 52 and 62. Global accuracy and Macro F1 do not improve.
- **Q18 — Verified with limitation:** `remoteSyncEnabled` is false, direct `syncPending()` calls are rejected, and local training creates no remote payload or pending synchronization record. Four focused tests passed in an isolated tracked-source copy. The full Flutter suite and unrelated network/export features were not tested.

## Current status table

| Question | Status | Actual output |
|---|---|---|
| Stage 1 | Verified with limitation | 26/26 research artefact hashes match. Python=e4aabb55a7d779e2e09dcac59c2a5746895c2149; Flutter=11daf49f491e70f9f7057cc3cf3bc3616990bb96; both main branches are 0 ahead/0 behind with no tracked or staged changes. |
| Q1 | Verified | Research contract is 105 features/13 classes; deployment contract is 4,096 features/14 classes. |
| Q2 | Verified with limitation | The fixed BudgetWise 'notes' column is read retrospectively; no participant-response collection exists in Tier A. |
| Q3 | Verified with limitation | Outputs are retrospective proxy/selection metrics; no participant usability dataset is present. |
| Q4 | Verified | train=7494, validation=1729, test=2308, training clients=150, labels=13. |
| Q5 | Verified with limitation | Proposed standard: accuracy=0.183709, Macro F1=0.043183, active classes=2.33, zero-recall=10.67. Class-weighted: accuracy=0.150202, Macro F1=0.066090, active classes=6.33, zero-recall=7.67. Train max/min canonical class counts=1623/184 (ratio 8.82). |
| Q6 | Verified with limitation | Overlapping transaction-ID values: train–validation=106; train–test=166; validation–test=40. Exact available-record overlaps=[0, 0, 0]; shared normalised note values=[24, 24, 24]. Transaction IDs are not globally unique, so repeated ID values alone do not establish duplicated transactions. No reliable merchant identifier or separate merchant-description field is available. |
| Q7 | Verified | Encoders, scaler and note TF-IDF are fitted in ProposedFeatureBuilder.fit(train); validation/test use transform_parts without refitting. |
| Q8 | Verified | ceil(0.30*2308)=693; every strategy/seed selected 693=True. |
| Q9 | Verified | Five strategies present; all select 693; all compare the same 2,308 IDs per seed=True. Three-seed recalculation matches the saved selector summary. |
| Q10 | Verified with limitation | Prompt precision=0.848004; source note used=1736/2079=0.835017. |
| Q11 | Verified with limitation | cases=1736, duplicates=0, wrong→correct=56, correct→wrong=54, changed=315. |
| Q12 | Verified with limitation | confidence_after > confidence_before and final prediction wrong: 874/1,736 (50.35%); stricter wrong before and after: 845/1,736 (48.68%). |
| Q13 | Verified | All 30 seed-round comparisons match=True; each method has 150 clients in every round. |
| Q14 | Verified | multiplier range=[0.912590, 0.931737], all within bounds=True; coefficients nonnegative=True; max \|sum-1\|=8.216e-15. |
| Q15 | Verified with limitation | Separate matched client-level audit: improved=0, declined=0, unchanged=450. Paired held-out predictions have zero category differences for seeds 42, 52 and 62; global accuracy and Macro F1 are identical per seed. |
| Q16 | Verified with limitation | sample-count/relative-adjustment correlation=0.095067; see quartile table for positive and negative adjustments in each size group. |
| Q17 | Verified with limitation | The implementation compares different model states, so the component combines representation and local-adaptation effects. |
| Q18 | Verified with limitation | remoteSyncEnabled=false; no remote upload payload or pending synchronization row; syncPending rejects direct calls. Four focused Flutter tests passed in an isolated tracked-source copy, including two optional-learning privacy tests. |
| Q19 | Verified | Research=105 features/13 classes; deployment=4096 features/14 classes; distinct=True. |
| Q20 | Verified with limitation | Class weighting raises proposed mean Macro F1 from 0.043183 to 0.066090 while mean accuracy falls from 0.183709 to 0.150202; full vs no-adjustment has identical accuracy/Macro F1 per seed. |

The CSV and Excel evidence matrices contain the detailed evidence paths, checks and required wording. Historical pre-freeze reports are preserved unchanged under `audit/historical/` and must not be treated as the current status.
