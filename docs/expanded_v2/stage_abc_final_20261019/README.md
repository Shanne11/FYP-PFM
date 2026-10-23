# Expanded V2 Stage A–B–C — public result evidence

**Source freeze:** `STAGE_ABC_LATEST_FREEZE_V1.zip` (23 October 2026 consolidated archive, including the 19 October 2026 Stage B/C closure).  
**Publication scope:** Sanitised, aggregate-only review package for the public FYP-PFM research repository.  
**Thesis location:** Chapter 5, Section 5.9 (Expanded V2 Genuine Smart Note Evaluation).

This folder is **not** the PocketIQ physical-device test package and **not** the primary Tier A federated-learning experiment. It documents a separate centralised transaction-categorisation study, using distinct data, models and evaluation protocols.

## Where to start

| Stage | Public evidence | Scope |
|---|---|---|
| Stage A | [Diagnostic reports and five-seed A3/A5 table](01_stage_a/) | Retrospective original bank descriptions; **not** newly written Smart Notes |
| Stage B | [B0/B1/B2 results and uncertainties](02_stage_b/) | Genuine owner-written Smart Notes, with B1/B2 using the same frozen model |
| Stage C | [Offline selector and gate comparison](03_stage_c/) | **Pre-collected-note replay**; no live prompting |
| Limitations | [Protocol, template overlap and figures](04_limitations/) | Generalisation and participant-data boundaries |
| Verification | [Final closure, numerical table and check results](05_verification/) | Saved-result verification, not retraining |
| Traceability | [PUBLIC_ARTIFACT_MANIFEST.csv](PUBLIC_ARTIFACT_MANIFEST.csv) | SHA-256 of each public file, with source archive path |

## Stage A — original bank-description context

The five-seed template-group-disjoint comparison reported mean Accuracy **0.3141** for A1 metadata-only and **0.5703** for A3 metadata plus word TF-IDF. The A5 equal-tuning model achieved mean Accuracy **0.6056** and Macro F1 **0.2639**, but its larger fitted representation gave it more predictive features. Under a matched text-feature budget, A5 achieved **0.5695** Accuracy and **0.2424** Macro F1: no stable character-feature advantage was established.

A0's near-zero mean accuracy (~0.04%) was independently traced to class-concentration and distribution shift induced by group-disjoint splitting, rather than a detected label-alignment defect. Stage A does **not** provide genuine new Smart Note evidence. Historical Stage A documents from earlier development stages describe Stage B/C as blocked; that statement was superseded when the separate participant study was completed and verified on 19 October.

## Stage B — genuine owner-written Smart Notes

Thirty participants contributed **254** owner-confirmed transaction records and answered notes. The five seeded evaluations are participant-disjoint **within each seed**. B1 and B2 use identical frozen models and preprocessing within each seed; only the presence of optional Smart Note text differs.

| Model | Mean Accuracy | Mean Macro F1 | What changes |
|---|---:|---:|---|
| B0 dedicated no-note | 0.6322 | 0.2949 | No-note task baseline |
| B1 same-model no-note | 0.6463 | 0.2966 | Unified model, note absent |
| B2 same-model with note | **0.7910** | **0.3862** | Identical frozen model, owner-written note present |

B1 → B2 yielded **32 Wrong-to-Correct, 1 Correct-to-Wrong, 137 Correct-to-Correct, 44 Wrong-to-Wrong**, or **+31 net corrections** over **214 transaction-seed cases**. These are **not 214 unique independent transactions** because test partitions across the five seeds can overlap. The findings support the benefit of genuine notes within this frozen participant dataset, not universal Smart Note effectiveness.

## Stage C — offline pre-collected-note replay

The comparison evaluates No Query, Random, Entropy and Margin under nominal 5%, 10%, 20% and 30% query budgets and the G0, G1 and G2 gate variants. The nominal **30% Margin/G1** configuration achieved a **31.31% realised mean query share**, **0.7491 mean Accuracy**, and **+22 net corrections** across the five seeded test evaluations. This was a **descriptive** result only; no statistical superiority over other selectors or live user acceptance is demonstrated. The Learned Benefit selector and G3 learned gate are **blocked/not evaluated** due to the inability to construct the required grouped out-of-fold targets without leakage.

## Limitations and privacy

- Reference labels were owner-confirmed, **not externally adjudicated**.
- Per-seed participant separation does not imply description-template separation; template overlaps were observed. Participants and templates formed one connected component, preventing component-disjoint sensitivity.
- Housing and Insurance had zero support; several observed categories were sparse.
- All 254 available notes were answered; skipped/non-response behaviours and per-note response times were not evaluated.
- Written institutional/ethics approval confirmation was not present in the **shareable** closure evidence; do not infer approval from package completeness.
- Public source files include only aggregate summaries, charts and saved-result verification records. The **private master freeze** is required for full custody of data and research files.
- Raw Google Forms data, notes, participant-level metrics/aliases, prediction-level records, original source workbook, split manifests, consent records, local environment paths and nested archives are **intentionally excluded** from this public selection. No identifiable transaction content is intended to be published.
- Stage B/C original training cannot be reproduced from this folder alone because restricted inputs and model packages remain outside the public archive. The reports describe checks against saved outputs.

**Do not compare** Stage A accuracy, Stage B accuracy, PocketIQ frozen ONNX deployment accuracy or Tier A federated-learning scores as if they were matched experimental conditions.

## Provenance

The selected tables and figures are reproduced **without modifying their values** from the final closure subarchive within `STAGE_ABC_LATEST_FREEZE_V1.zip`. Stage A diagnostics come from its preserved `02_STAGE_A_FINAL/` path. See [PUBLIC_ARTIFACT_MANIFEST.csv](PUBLIC_ARTIFACT_MANIFEST.csv) for filenames, hash digests and original paths. The private freeze is intentionally **not committed** here.
