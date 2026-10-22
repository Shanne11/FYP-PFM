# BudgetWise Tier A V2 — Exploratory Research

This directory contains the exploratory 108-feature
BudgetWise federated experiments.

**It is not the authoritative 105-feature Tier A experiment
and is not the 30-participant Expanded V2 Stage B/C study.**

[Return to experiment registry](../../docs/EXPERIMENT_REGISTRY.md)

## Experiment contract

| Setting | Value |
| --- | --- |
| Source | BudgetWise synthetic finance dataset |
| Cleaned records | 14,011 |
| Train / validation / test | 9,108 / 2,102 / 2,801 |
| Split | Frozen transaction-ID group-safe |
| Simulated clients | 150 |
| Input features | 71 metadata + 37 notes = 108 |
| Categories | 13 |
| Rounds | 40 |
| Local epochs | 3 |
| Training seed | 42 |
| Class-balanced loss | Effective number, beta 0.9999 |
| Held-out test scored | 0 |

Feature preprocessing is fitted using training data only.

## Main validation results

| Configuration | Macro F1 |
| --- | ---: |
| FedAvg | 0.130460 |
| FedProx | 0.129420 |
| Proposed, bounded aggregation | 0.125830 |
| Proposed, neutral aggregation | 0.125460 |
| Filtered notes, neutral aggregation | 0.126293 |
| Filtered notes, bounded aggregation | 0.125919 |

These are seed-42 validation results, not the primary
three-seed Tier A thesis comparisons.

The two filtered experiments exclude predefined placeholder
notes without using true category labels as a filtering rule.

The improvements are small. Filtering does not demonstrate
superiority over the V2 FedAvg baseline.

## Smart Note diagnostic findings

The retrospective source notes are not participant-authored
answers to ACTM prompts.

The training-note catalogue contains only 26 distinct
nonempty note texts. All 26 occur in all 13 categories.

Some semantic mismatches already exist in the original
BudgetWise CSV. This does not independently prove that
the original transaction category labels are wrong.

The same-checkpoint note quality diagnostic recorded:

| Finding | Count |
| --- | ---: |
| ACTM-selected prompt slots | 631 |
| Initially wrong predictions targeted | 497 |
| Selected notes used | 528 |
| Wrong to correct | 5 |
| Correct to wrong | 15 |
| Net corrections | -10 |

In 50 shuffled-note permutations, 82% produced Macro F1
at least as high as the original note assignments.

This supports a note-alignment limitation, not a causal
claim about how the source notes were generated.

## Available scripts

- `build_dataset_v2.py` — cleaning and source audit.
- `fix_split_v2.py` — versioned split preparation.
- `prepare_features_v2.py` — V2 feature contract.
- `run_validation_v2.py` — original V2 methods.
- `diagnose_actm_selection_v2.py` — prompt selection.
- `diagnose_actm_trigger_quality_v2.py` — selection quality.
- `diagnose_note_quality_v2.py` — note and category outcomes.
- `diagnose_note_permutation_v2.py` — shuffled-note control.
- `run_note_quality_filter_v2.py` — placeholder filter.
- `diagnose_central_v2.py` — centralised control.

## Published results

Original results:
[Validation summaries](outputs/validation/full/)

Exploratory diagnostics:
[Diagnostic summaries](outputs/diagnostics/)

Source audit:
[Dataset audit](reports/dataset_v2_audit.json)

Feature dimensions:
[Feature contract](reports/feature_contract_v2.json)

Note quality filtering:

- [Neutral summary](outputs/validation/full/proposed_note_quality_placeholder_neutral_seed_42/summary.json)
- [Bounded summary](outputs/validation/full/proposed_note_quality_placeholder_bounded_seed_42/summary.json)

## Reproducibility boundary

The cleaned V2 CSV is generated from the preserved
BudgetWise source dataset. The source, split manifest
and scripts must match the original data fingerprint.

GitHub does not contain every round-metrics CSV,
full prediction file or best-model checkpoint.

Preserve original local artifacts and keep outputs
from new experiments in separate versioned directories.

## Interpretation

BudgetWise V2 demonstrates promising ACTM error targeting,
but its retrospective notes are poorly aligned with
transaction categories. Placeholder filtering only
slightly improves validation results.

No substantial beneficial effect of bounded heuristic
aggregation is established.

Genuine-note findings belong to the separately documented
Expanded V2 Stage B/C evaluation.