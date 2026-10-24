# Expanded V2 Stage B/C Methodology Verification

**Project:** Clarification-Aware Federated Transaction Categorisation for Personal Finance with Smart Notes  
**Prepared:** 24 October 2026  
**Evidence basis:** User-supplied, read-only Codex audit of the original private Expanded V2 experiment source, saved configurations and verified archive. This document reorganises those findings for thesis traceability; its author has not independently reopened the restricted source archive.  
**Research boundary:** Expanded V2 is a **separate centralised Python study**. Its results are not the primary 105-feature/13-category Tier A federated results and are not performance results for PocketIQ's 4,096-feature/14-category ONNX model.

## 1. Provenance and scope

The local audit reported byte-for-byte matches for ten core Stage B/C source files against the verified experiment archive, with no file edits, retraining or reruns. The original verified archive was `STAGE_BC_EXPANDED_V2_FINAL_VERIFIED_2026-10-19.zip`, SHA-256:

`536e5db233ba8c4c752e59ea85622c40c2d07739106626aa75c73acc942e9dd0`

Additional preservation archives were reported, including the final Stage B/C closure (SHA-256 `9fc80e52b3c8da95ea5094534592477e235a38177ac268e8168d5e822076d907`) and consolidated Stage A/B/C freeze (SHA-256 `bf2b207c6fc8a8580be9a5d10422057bb7c8d005e9c50ddbb7b51b7683a17565`). These archives confirm preservation, not an embedded, authoritative model-training timestamp.

No raw responses, participant identifiers, descriptions, owner-written note text, private workbook extracts, consent/withdrawal data or fitted model binaries are contained in this public-facing methodology record.

## 2. Data and participant-disjoint split construction

The expanded study comprised **30 participants and 254 owner-confirmed transaction records**, under a fixed **14-category** label universe. Evaluations used seeds **42, 52, 62, 72 and 82**.

For each seed, `participant_only_split()` used `GroupShuffleSplit(train_size=0.70, random_state=seed)` to assign 21 participants to training. A second `GroupShuffleSplit(train_size=0.50, random_state=seed+1000)` divided the nine remaining participants into four validation and five test participants. No participant appeared in more than one partition within a seed, and labels were not used for split construction.

| Seed | Training records | Validation records | Test records | Participants (train/val/test) |
|---|---:|---:|---:|---|
| 42 | 176 | 35 | 43 | 21 / 4 / 5 |
| 52 | 178 | 35 | 41 | 21 / 4 / 5 |
| 62 | 178 | 32 | 44 | 21 / 4 / 5 |
| 72 | 178 | 32 | 44 | 21 / 4 / 5 |
| 82 | 177 | 35 | 42 | 21 / 4 / 5 |

**Limit:** The splits are *participant-disjoint* but not description-template- or merchant-disjoint. Identical description groups occur in training/validation and held-out records. Test records and participants may reappear across the five independently seeded splits; the 214 held-out transaction-seed evaluations are not independent unique transactions.

**Original source index:** `prepare_expanded_v2.py` (`participant_only_split`, approximately line 91); `run_stage_b_expanded_v2.py` (split override, approximately line 51); per-seed `split_manifest.csv`.

## 3. Model families, features and missing-data handling

### 3.1 Common model settings

Stage B used scikit-learn 1.5.1 `LogisticRegression` with `solver='lbfgs'`, `max_iter=500`, experiment-seed `random_state`, regularisation `C` chosen from `{0.1, 1.0, 10.0}` and class weighting chosen from `{None, 'balanced'}`. The `multi_class` argument was not explicitly set (the installed library's default behavior applied).

Metadata included transaction type, debit amount, credit amount, derived amount direction, and missing-value indicators. Transaction type and amount direction were one-hot encoded with unknown values ignored. Amounts underwent a signed `log1p(abs(value))` transform and training-fitted `StandardScaler(with_mean=False)`. Invalid/missing amounts were represented by zero after transformation plus an explicit missing indicator. Missing text was represented by an empty string. Text normalisation applied Unicode NFKD and lowercase conversion.

Description text used **word TF-IDF unigrams/bigrams** (`max_features=4000`, `sublinear_tf=True`); Smart Notes used a separate **word TF-IDF unigrams/bigrams** (`max_features=2000`, `sublinear_tf=True`). Metadata transformers, scalers and both vocabularies were fitted on training records only; unspecified TF-IDF properties followed scikit-learn 1.5.1 defaults. These maxima are not claims about the exact number of nonzero/fitted features in each seed.

**Original source index:** `src/smartnote_abc/features.py` (`MetadataTransformer`, approximately line 20); `src/smartnote_abc/notes.py` (`UnifiedOptionalNoteModel`, approximately lines 274–282).

### 3.2 B0, B1 and B2

- **B0:** A separately trained, dedicated no-note logistic-regression classifier, using metadata and the original transaction description, but no Smart Note vectorizer or missing-note indicator.
- **B1:** The unified optional-note classifier evaluated with `note_text=''` and `note_missing=1`.
- **B2:** **The same unified classifier as B1** evaluated with an available owner-written note and the corresponding recalculated missing-note indicator. Model coefficients, intercept, feature schema, fitted vocabularies, metadata preprocessing and category order remain unchanged between the paired B1/B2 inferences.

For unified-model training, each answered training transaction was represented with note-present and note-absent variants weighted **0.5** each (`note_dropout=0.5`). These weighted variants preserve total weight one for the original transaction. B0 hyperparameters were selected by validation Macro F1 and then validation accuracy. Unified-model selection prioritised **B2 validation Macro F1**, then **B1 validation Macro F1**, then **B2 validation accuracy**. Further ties favoured smaller `C`, then unweighted over `balanced`. Selected classifiers were fitted on training variants only; validation observations were not added to final fitting.

| Seed | B0 selected | Unified B1/B2 selected |
|---|---|---|
| 42 | `C=1`, balanced | `C=10`, balanced |
| 52 | `C=10`, balanced | `C=10`, balanced |
| 62 | `C=10`, balanced | `C=10`, balanced |
| 72 | `C=10`, balanced | `C=10`, balanced |
| 82 | `C=10`, balanced | `C=10`, balanced |

Probability columns were aligned to the frozen 14-label ordering; missing training classes were given zero probability columns and each row was checked to sum to one. Paired B1/B2 model fingerprints were verified equal for all seeds.

**Original source index:** `src/smartnote_abc/notes.py` (`build_note_training_variants`, approximately line 236; `predict_pair`, approximately line 311; `DedicatedXModel`, approximately line 332); `run_same_model_notes.py` (selection logic, approximately lines 140 and 166); `src/smartnote_abc/metrics.py` (`aligned_probabilities`, approximately line 15); per-seed `selected_config.json`.

## 4. Stage C offline selection rules

Stage C is **offline pre-collected-note replay**, not live ACTM prompting. Selection used no-note (B1) information prior to revealing any stored Smart Note. It compared Random, Entropy and Margin at nominal budgets of 5%, 10%, 20% and 30%, calculated *per seed* as:

\[
q=\lceil N_{\mathrm{test}}b\rceil.
\]

This ceiling means the realised proportion can exceed the nominal proportion: at the nominal 30% level, 67 of 214 held-out transaction-seed cases were queried (**31.31%**), so this is **not a strict 30%-maximum realised share**.

**Entropy:** Rank descending by normalised Shannon entropy,

\[
H(x)=\frac{-\sum_{c=1}^{K} p_c(x)\ln p_c(x)}{\ln K}.
\]

Probabilities are clipped to machine epsilon before the logarithm.

**Margin:** Rank ascending by the raw top-two-class probability margin,

\[
M(x)=p_{(1)}(x)-p_{(2)}(x).
\]

**Random:** Generate one random score per test record using `np.random.default_rng(seed).random(number_of_test_records)` and rank scores descending.

**Tie-handling:** Sorting is stable (`mergesort`), with a deterministic record-specific secondary tie value from `default_rng(seed + sum(ord(character) for character in record_id))`, followed by lexicographical pseudonymous record ID. Primary score is descending for Entropy and Random, ascending for Margin.

**Original source index:** `src/smartnote_abc/stage_c.py` (`normalized_entropy`, approximately line 58; `top_two_margin`, approximately line 65; selection scoring, approximately lines 239–245); `src/smartnote_abc/selectors.py` (`select_budget`, approximately line 22); Stage C `run_configuration.json`.

## 5. Stage C gate definitions and G2 tuning

- **G0:** Retain the no-note (B1) prediction.
- **G1:** Use note-enhanced (B2) prediction when a selected, answered and non-empty note is available.
- **G2:** Use note-enhanced prediction only if a note is available, its `new_purpose_information` score meets a validation-selected threshold, **and** note-enhanced model confidence is at least the no-note confidence.

`new_purpose_information` is the share of **distinct lowercased alphanumeric tokens in the note that are absent from the original transaction description**. This is a lexical novelty feature, not a validated semantic measure of human purpose.

For each seed, G2 tested thresholds `{0.00, 0.25, 0.50, 0.75, 1.00}`. The selection priority was **validation Macro F1**, followed by **validation accuracy**, then the **larger threshold**. Held-out test labels were not used for threshold selection.

| Seed | Selected G2 threshold | Validation Macro F1 | Validation accuracy |
|---|---:|---:|---:|
| 42 | 0.50 | 0.392768 | 0.771429 |
| 52 | 1.00 | 0.235931 | 0.742857 |
| 62 | 1.00 | 0.261547 | 0.812500 |
| 72 | 1.00 | 0.348622 | 0.937500 |
| 82 | 1.00 | 0.205383 | 0.714286 |

For seeds 52, 62, 72 and 82, the candidate thresholds tied and the larger-threshold rule selected 1.00. For seed 42, 0.00, 0.25 and 0.50 tied at the best two metrics, and 0.50 was selected.

**Original source index:** `src/smartnote_abc/gating.py` (`apply_gate`, approximately line 21; threshold selection, approximately line 92); `src/smartnote_abc/stage_c.py` (`gate_features`, approximately line 89); `run_stage_c_expanded_v2.py` (threshold selection, approximately line 124).

**Not evaluated:** The Learned Benefit selector and G3 learned gate, because participant/description linkage left one connected component and prevented the required leakage-safe grouped out-of-fold training.

## 6. Leakage boundaries and verification

- Training, validation and test **participant identities are disjoint within each seed**.
- Target-bearing transaction columns are excluded from input features.
- Metadata preprocessing and both TF-IDF vocabularies use training-only fitting.
- Hyperparameters and G2 gates use validation information; test outcomes are evaluated afterward.
- Stage C query selection excludes note text/status, reference labels, post-note predictions and prediction transitions. Query features are base confidence, entropy, margin and metadata missingness.
- Note text is revealed only after query selection for gated offline replay.
- Run provenance stores partition IDs and vocabulary hashes; paired inference and replay check unchanged model fingerprints.

**Limitations:** Neither template overlap nor independent participant samples across the five repeated seeds are eliminated. Selecting a note after offline precollection does not measure real-time prompt acceptance, effort or writing time.

## 7. Freeze chronology and reproducibility limitation

The read-only audit recovered the following **filesystem/archive chronology** in Asia/Kuala_Lumpur local time:

| Evidence item | Recorded metadata time |
|---|---|
| Stage B model artifacts | Approximately 19 October 2026, 16:30 |
| Stage B run configuration | 19 October 2026, 16:30:54 |
| Stage C model artifacts | Approximately 19 October 2026, 16:31 |
| Stage C run configuration | 19 October 2026, 16:31:46 |
| Verified ZIP | 19 October 2026, 18:13:30 |
| Closure build | 19 October 2026, 21:36 |
| Consolidated Stage A–B–C preservation archive | 19 October 2026, 22:19 |

**Important qualification:** No authoritative per-model or per-G2-record `frozen_at` property was found. The exact wall-clock time of each model fit was not retained. Filesystem and archive times must **not** be presented as definitive pre-evaluation freeze timestamps. Supported reproducibility statements concern validation-only model/gate selection, archived immutable configurations, recorded checksums and agreement of paired model fingerprints. The Stage C runner deterministically fitted its own copy of the Stage B model rather than loading Stage B's joblib directly; the saved copies matched byte-for-byte per seed.

## 8. Open governance / evidence items

**Unresolved: Institutional ethics or exemption documentation.** The inspected shareable archive referred to student-reported supervisor-delegated approval but did not include an independently verifiable institutional approval or exemption instrument. Obtain the applicable institution/faculty record (if one exists) or a documented determination of exemption/not-required; report only the actual status. Active consent entries are not equivalent to documented institutional approval.

**Additional known limitations:** The archival package omits restricted raw participant sources. Category definitions/exclusion examples require independent documentation where appropriate. Stage C does not measure live response behavior or individual note-writing times.

## 9. Safe publication and thesis use

This Markdown file may be committed as a **non-sensitive provenance summary** under `docs/expanded_v2/EXPANDED_V2_METHODOLOGY_VERIFICATION.md` after checking it for local confidentiality requirements. Do **not** commit the original verified/closure ZIPs, raw source spreadsheets, participant-level data, consent registries, fitted binaries or private local-drive paths.

Use the factual methodology in Chapter 3 (a dedicated Expanded V2 experimental protocol subsection is preferable), retain Chapter 5's saved numerical results without rerunning, and cite the existing Appendix C diagnostics for template overlap and clustered uncertainty. Do not mark Chapter 3's ethics-review finding fully resolved until institutional governance has been confirmed.
