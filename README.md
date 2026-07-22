# Personal Finance Management AI Framework

This repository implements and evaluates a privacy-aware transaction categorisation framework for a multi-account Personal Finance Management system. It compares rules, centralised machine learning, federated learning, and a proposed human-in-the-loop method based on ACTM, selective Smart Notes, note utility, and bounded utility-weighted aggregation.

## Current research status

The repository contains six executable methods:

| ID | Method | Learning setting | Status |
|---|---|---|---|
| B1 | Rules-only | Centralised heuristic | Implemented and rerun |
| B2 | Metadata-only Random Forest | Centralised | Implemented and rerun |
| B3 | Metadata + Notes Random Forest | Centralised | Implemented and rerun |
| B4 | FedAvg MLP | Federated | Implemented and rerun |
| B5 | FedProx MLP | Federated | Implemented and rerun |
| P | ACTM + selective Smart Notes + note utility + bounded utility-weighted FedAvg | Federated, human-in-the-loop | Implemented and rerun |

The corrected implementation, three-seed federated evaluation, and proposed-method ablation study have been completed. The results support ACTM as the clearest contribution, but they do **not** show a material benefit from the current utility-weighted aggregation. Ambiguous-subset and prompt-efficiency analysis remain necessary before final Chapter 5 reporting.

## Corrected experiment contract

All methods use the same data contract defined by [`utils/experiment_data.py`](utils/experiment_data.py) and [`data/experiment_split.json`](data/experiment_split.json).

### Canonical categories

The source dataset contains 31 inconsistent labels. They are normalised into 13 categories:

- Bonus
- Education
- Entertainment
- Food
- Freelance
- Health
- Investment
- Other
- Rent
- Salary
- Savings
- Travel
- Utilities

For example, `Food`, `FOOD`, `Foodd`, `Fod`, `Foods`, and `food` represent the same `Food` category. Unknown category spellings cause the experiment to stop instead of silently creating another output class.

### Frozen split

The versioned split manifest contains:

| Partition | Records | Purpose |
|---|---:|---|
| Training | 7,494 | Model fitting and federated client training |
| Validation | 1,729 | Federated checkpoint selection |
| Test | 2,308 | Final held-out evaluation |

Every method uses the exact same source rows. The manifest verifies the dataset fingerprint, checks that the partitions are disjoint, and confirms complete coverage. Source-row indices are used because the supplied dataset contains duplicate transaction IDs.

Encoders, scalers, and note vocabularies are fitted on training records only. Federated clients are also constructed exclusively from the training partition, preventing test-data leakage.

## Proposed method

The proposed experiment follows this pipeline:

```text
Metadata-only local prediction
        |
        v
ACTM uncertainty decision
  - predictive entropy
  - top-two probability margin
  - cross-context conflict
  - prompt budget
        |
        v
Selective Smart Note fusion
        |
        v
Smart Note utility
  - uncertainty reduction: 0.50
  - semantic specificity: 0.30
  - bounded effort: 0.20
        |
        v
Local client training
        |
        v
Sample-count FedAvg weight
  x utility multiplier [0.75, 1.25]
        |
        v
Normalised global aggregation
```

If a client has missing, invalid, or insufficient note utility, its multiplier returns to `1.0`. The final model is selected using validation Macro F1, reloaded from the best checkpoint, and evaluated once on the held-out test set.

### Dataset limitation

The dataset does not provide merchant or account columns. The current conflict detector therefore uses:

- `location` as the merchant-context proxy;
- `payment_mode` as the account/channel proxy.

This must be described as a limitation. It should not be presented as an evaluation using genuine merchant and account identifiers.

## Repeated-seed federated results

FedAvg, FedProx, and Proposed were evaluated using training seeds 42, 52, and 62 while keeping the corrected dataset and frozen split unchanged. Values are mean +/- sample standard deviation across three runs.

| Method | Accuracy | Macro F1 | Weighted F1 | ECE | Brier score |
|---|---:|---:|---:|---:|---:|
| FedAvg | 0.1805 +/- 0.0094 | 0.0427 +/- 0.0065 | 0.0938 +/- 0.0092 | 0.0852 +/- 0.0081 | 0.9114 +/- 0.0064 |
| FedProx | 0.1798 +/- 0.0104 | 0.0424 +/- 0.0062 | 0.0933 +/- 0.0099 | 0.0847 +/- 0.0081 | 0.9114 +/- 0.0064 |
| Proposed | **0.1830 +/- 0.0141** | **0.0430 +/- 0.0066** | **0.0951 +/- 0.0127** | **0.0839 +/- 0.0088** | **0.9097 +/- 0.0096** |

The proposed method has the strongest mean federated accuracy, Macro F1, Weighted F1, ECE, and Brier score. However, the differences are small relative to the between-seed variation. These results should be described as modest improvements, not proof of a large or statistically established advantage.

## Proposed-method ablation results

The full method and six ablations were evaluated with the same seeds. Values are mean +/- sample standard deviation.

| Variant | Accuracy | Macro F1 | Weighted F1 | ECE | Brier score |
|---|---:|---:|---:|---:|---:|
| Full | **0.1837 +/- 0.0149** | 0.0432 +/- 0.0058 | **0.0951 +/- 0.0149** | 0.0847 +/- 0.0061 | 0.9097 +/- 0.0097 |
| Without ACTM | 0.1804 +/- 0.0096 | 0.0426 +/- 0.0060 | 0.0930 +/- 0.0108 | 0.0856 +/- 0.0062 | 0.9115 +/- 0.0064 |
| Without notes | 0.1826 +/- 0.0111 | **0.0438 +/- 0.0073** | 0.0944 +/- 0.0131 | **0.0836 +/- 0.0088** | **0.9097 +/- 0.0097** |
| Simple concatenation | 0.1830 +/- 0.0141 | 0.0430 +/- 0.0066 | 0.0951 +/- 0.0127 | 0.0839 +/- 0.0088 | 0.9097 +/- 0.0096 |
| Without uncertainty utility | 0.1836 +/- 0.0147 | 0.0432 +/- 0.0058 | 0.0950 +/- 0.0148 | 0.0846 +/- 0.0062 | 0.9097 +/- 0.0097 |
| Without specificity utility | 0.1836 +/- 0.0147 | 0.0432 +/- 0.0058 | 0.0950 +/- 0.0148 | 0.0846 +/- 0.0062 | 0.9097 +/- 0.0097 |
| Without utility weighting | 0.1837 +/- 0.0149 | 0.0432 +/- 0.0058 | 0.0951 +/- 0.0149 | 0.0847 +/- 0.0061 | 0.9097 +/- 0.0097 |

### Findings supported by the ablation study

- ACTM provides the clearest positive contribution: removing it reduces mean accuracy by 0.0033, Weighted F1 by 0.0021, and worsens Brier score by 0.0018.
- Selective note use gives small accuracy and Weighted F1 gains, but the no-notes variant has slightly higher Macro F1 and better ECE. The note effect is therefore mixed.
- Semantic-anchor-gated fusion gives only a very small classification improvement over simple concatenation and does not improve calibration in these runs.
- Removing either uncertainty-reduction utility or semantic-specificity utility changes the results only negligibly.
- Removing utility weighting produces effectively identical results to the full method. The current evidence does not support a claim that utility-weighted aggregation materially improves performance.
- Federated Macro F1 remains low, showing that minority-category performance is still weak despite the modest aggregate improvement.

The null utility-weighting result is retained as an honest research finding. The diagnostic below explains why the mechanism currently has negligible aggregation impact.

### Utility-weighting diagnosis

The three full-method runs were diagnosed across 4,500 client-round observations and 57,208 selected-note observations:

| Diagnostic | Result |
|---|---:|
| Client mean utility | 0.3252–0.3635 |
| Utility multiplier | 0.9126–0.9317 |
| Multiplier standard deviation | 0.0026 |
| Within 0.01 of median multiplier | 100.0% |
| Fallback to 1.0 | 0.0% |
| Mean absolute base/final weight change | 0.0000133 |
| Mean relative base/final weight change | 0.20% |

Utility-component distributions show why the client means are compressed:

| Component | Mean | Standard deviation | Median | Range |
|---|---:|---:|---:|---:|
| Uncertainty reduction | 0.0056 | 0.0074 | 0.0027 | 0.0000–0.0641 |
| Semantic specificity | 0.9199 | 0.2715 | 1.0000 | 0.0000–1.0000 |
| Bounded effort | 0.1886 | 0.0880 | 0.2500 | 0.0000–0.3750 |
| Combined transaction utility | 0.3165 | 0.0938 | 0.3500 | 0.0000–0.4012 |

Specificity is saturated at `1.0` for most selected notes, uncertainty reduction is close to zero, and averaging many notes compresses client-level utility further. Consequently, nearly uniform multipliers are removed again by final weight normalisation and have negligible aggregation impact.

**Verdict:** the current utility-weighted aggregation is an unsupported/negative finding. It must not be claimed as a demonstrated performance improvement. A revised mapping or specificity measure may be investigated using development data, but it must then be rerun on frozen evaluation settings and compared across seeds before any positive claim is made.

### ACTM and prompt-efficiency diagnosis

The three full-method runs produce the following ACTM evidence:

| Metric | Three-seed result |
|---|---:|
| Ambiguity eligibility rate | 100.0% |
| Prompts per 100 transactions | 30.03 |
| Prompt precision | 84.80% +/- 1.06% |
| Note acceptance rate | 83.50% +/- 0.68% |
| Mean uncertainty reduction | 0.000276 +/- 0.000268 |
| Ambiguous-subset Macro F1 | 0.0432 +/- 0.0058 |

Every held-out transaction satisfies at least one ambiguity condition. Consequently, the “ambiguous subset” is the complete test set and its metrics are identical to the overall proposed-method metrics. Selectivity is currently produced by the 30% prompt budget rather than by the entropy, margin, and conflict thresholds separating a smaller ambiguous subset.

Prompt precision and note acceptance are high, showing that ACTM can prioritise likely errors within a fixed interaction budget. However, uncertainty reduction after clarification is negligible. ACTM should therefore be reported as evidence for **budgeted prioritisation**, not as a validated ambiguity separator or uncertainty-reduction mechanism. Threshold calibration must be performed on validation data if a genuinely selective ambiguous subset is required.

### Federated class-level diagnosis

The low federated Macro F1 is caused by prediction collapse rather than uniformly weak performance across all categories:

| Method | Best mean class F1 | Categories with zero recall in all seeds | Categories never predicted in all seeds |
|---|---:|---:|---:|
| FedAvg | Food: 0.2477 | 9/13 | 8/13 |
| FedProx | Food: 0.2456 | 9/13 | 8/13 |
| Proposed | Food: 0.2689 | 9/13 | 9/13 |

For Proposed, the mean prediction shares are approximately 41.3% Food, 42.7% Rent, 14.8% Utilities, and 1.2% Investment. Bonus, Education, Entertainment, Freelance, Health, Other, Salary, Savings, and Travel have zero recall in every seed. The largest aggregate confusion pairs are Food→Rent (643), Rent→Food (541), and Travel→Food (382).

Although the held-out support ranges from 57 Savings records to 500 Food records, the model’s predictions are substantially more concentrated than the underlying class distribution. Accuracy and Weighted F1 are therefore dominated by a few larger categories and overstate practical 13-category performance. Macro F1 is the appropriate primary metric for the multiclass claim, and the current federated models should be described as suffering from severe class collapse.

## Single-seed reference results

The following results were generated using the corrected 13-category dataset and shared split with seed 42:

| Method | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Rules-only | 0.0368 | 0.0432 | 0.0205 | 0.0270 | 0.0501 |
| Metadata-only | 0.3011 | 0.1474 | 0.1674 | 0.1381 | 0.2120 |
| Metadata + Notes | **0.3120** | **0.2584** | **0.1856** | **0.1582** | **0.2201** |
| FedAvg | 0.1876 | 0.0306 | 0.0772 | 0.0413 | 0.1044 |
| FedProx | 0.1880 | 0.0307 | 0.0774 | 0.0414 | 0.1047 |
| Proposed | **0.1967** | 0.0294 | 0.0755 | **0.0423** | **0.1097** |

Bold values mark the strongest overall result or the strongest federated result, as applicable.

### Interpretation

- Metadata + Notes is currently the strongest overall classifier.
- Proposed is the strongest federated method in accuracy, Macro F1, and Weighted F1.
- Compared with FedAvg, Proposed improves accuracy by approximately 0.0091 and Macro F1 by approximately 0.0010.
- Compared with FedProx, Proposed improves accuracy by approximately 0.0087 and Macro F1 by approximately 0.0009.
- The federated improvement is positive but small. The current evidence does not support a claim that Proposed outperforms every baseline.
- Low federated Macro F1 indicates that minority-category performance remains weak and should be examined through class-level reports and confusion matrices.

These seed-42 values are retained for traceability. The repeated-seed results above should be used for federated method conclusions.

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/Shanne11/FYP-PFM.git
cd FYP-PFM

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, either adjust the current-user execution policy or call `.venv\Scripts\python.exe` directly.

### Linux or macOS

```bash
git clone https://github.com/Shanne11/FYP-PFM.git
cd FYP-PFM

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the experiments

Run the six methods from the repository root:

```powershell
python train_baseline.py
python train_metadata.py
python train_notes.py
python train_fedavg.py
python train_fedprox.py
python train_proposed.py
```

The Random Forest experiments normally finish first. FedAvg, FedProx, and Proposed take longer because they train across all simulated clients for 10 communication rounds and 3 local epochs.

After all six `overall_metrics.csv` files exist, generate the final comparison:

```powershell
python evaluation/compare_all_models.py
```

The comparison is written to:

```text
outputs/comparison/baseline_comparison.csv
outputs/comparison/baseline_comparison.png
```

### Repeated-seed federated evaluation

Run FedAvg, FedProx, and Proposed with training seeds 42, 52, and 62 while
keeping the frozen data split unchanged:

```powershell
python evaluation/run_repeated_seeds.py
```

This performs nine full federated runs and may take considerably longer than a
single comparison. Results are written to:

```text
outputs/repeated/repeated_runs.csv
outputs/repeated/repeated_summary.csv
outputs/repeated/*_mean_std.png
```

The summary reports the arithmetic mean and sample standard deviation. For a
non-reportable smoke test only, use:

```powershell
python evaluation/run_repeated_seeds.py --seeds 42 --rounds 1 --local-epochs 1 --max-clients 3
```

### Utility-weighting diagnostics

After completing the three full ablation runs, audit utility components, client multipliers, fallback frequency, and aggregation-weight changes:

```powershell
python evaluation/diagnose_utility.py
```

This creates:

```text
outputs/utility_diagnostics/client_round_diagnostics.csv
outputs/utility_diagnostics/utility_diagnostic_summary.csv
outputs/utility_diagnostics/utility_component_summary.csv
outputs/utility_diagnostics/diagnostic_conclusion.txt
outputs/utility_diagnostics/utility_diagnostics.png
```

### ACTM evaluation summary

Summarize ambiguous-subset and prompt-efficiency evidence across the three full runs:

```powershell
python evaluation/summarize_actm.py
```

This creates:

```text
outputs/actm_evaluation/actm_runs.csv
outputs/actm_evaluation/actm_summary.csv
outputs/actm_evaluation/actm_conclusion.txt
outputs/actm_evaluation/actm_rates.png
```

### Federated class-level analysis

Generate per-class precision, recall, F1, prediction shares, and aggregate confusion pairs from the repeated-seed predictions:

```powershell
python evaluation/analyze_federated_classes.py
```

This creates:

```text
outputs/class_analysis/class_runs.csv
outputs/class_analysis/class_summary.csv
outputs/class_analysis/confusion_pairs.csv
outputs/class_analysis/class_analysis_conclusion.txt
outputs/class_analysis/class_f1_comparison.png
```

## Proposed experiment options

Display available settings:

```powershell
python train_proposed.py --help
```

Important defaults:

| Setting | Default |
|---|---:|
| Communication rounds | 10 |
| Local epochs | 3 |
| Entropy threshold | 0.65 |
| Top-two margin threshold | 0.15 |
| Prompt budget | 0.30 |
| Utility multiplier range | 0.75-1.25 |
| Random seed | 42 |

`--max-clients` is intended only for development smoke tests. Do not use it for final experimental results.

## Ablation study

Run the full method and six component ablations using seeds 42, 52, and 62:

```powershell
python evaluation/run_ablations.py
```

The study evaluates:

- full proposed method;
- without ACTM (notes always available);
- without Smart Notes;
- simple note concatenation instead of semantic-anchor-gated fusion;
- without uncertainty-reduction utility;
- without semantic-specificity utility;
- without utility-weighted aggregation.

This performs 21 full federated runs. Outputs include raw runs, mean plus sample
standard deviation, deltas from the full method, and error-bar charts:

```text
outputs/ablations/ablation_runs.csv
outputs/ablations/ablation_summary.csv
outputs/ablations/ablation_deltas.csv
outputs/ablations/*_ablation.png
```

For orchestration testing only:

```powershell
python evaluation/run_ablations.py --seeds 42 --rounds 1 --local-epochs 1 --max-clients 3
```

## Generated outputs

Each experiment writes to its own directory:

```text
outputs/
|-- baseline1/
|-- baseline2/
|-- baseline3/
|-- baseline4/
|-- baseline5/
|-- proposed/
`-- comparison/
```

Depending on the method, artifacts include:

- `overall_metrics.csv`
- `classification_report.txt` or `metrics.txt`
- `confusion_matrix.png`
- `round_metrics.csv`
- `client_metrics.csv`
- `aggregation_weights.csv`
- `utility_scores.csv`
- `prompt_metrics.csv`
- `calibration_metrics.csv`
- `predictions.csv`
- best-model checkpoints
- fitted preprocessing pipelines

Model files, prediction-level CSV files, and datasets are ignored by default where appropriate. Aggregate evaluation files are allowed so results can be reviewed without committing large or sensitive artifacts.

## Repository structure

```text
FYP-PFM/
|-- data/
|   |-- build_experiment_split.py
|   `-- experiment_split.json
|-- dataset/
|-- evaluation/
|   `-- compare_all_models.py
|-- models/
|   `-- federated/
|-- outputs/
|-- tests/
|-- utils/
|   |-- experiment_data.py
|   |-- federated_baseline.py
|   |-- metrics.py
|   `-- proposed_features.py
|-- train_baseline.py
|-- train_metadata.py
|-- train_notes.py
|-- train_fedavg.py
|-- train_fedprox.py
|-- train_proposed.py
|-- requirements.txt
`-- README.md
```

## Validation

The current checks cover:

- complete mapping from all raw labels to 13 canonical categories;
- frozen split disjointness and full dataset coverage;
- presence of all categories in every partition;
- ACTM trigger reasons and prompt-budget enforcement;
- training-only cross-context conflict learning;
- sample-based aggregation weights, bounded utility multipliers, and fallback behaviour;
- syntax compilation and end-to-end federated smoke execution.

## Required next evaluation work

Before treating the results as final research evidence:

1. If utility weighting is revised, define the new mapping and specificity measure using development data only, then rerun the frozen three-seed evaluation and ablations.
2. Evaluate alternative prompt budgets and multiplier bounds without selecting settings using the final test results.
3. Calibrate ACTM thresholds on validation data if the report requires a genuinely selective ambiguous subset rather than budget-based ranking.
4. Investigate class-balanced loss, client sampling, or other training-only remedies for federated class collapse without changing the held-out test set.
5. Add uncertainty estimates or statistical testing appropriate for the final comparison claims.
6. Freeze the selected model and preprocessing contract before mobile integration.

## Scope

This repository is the research and model-development environment. It is not the complete Flutter Personal Finance Management application. The selected final model, category mapping, ACTM configuration, and preprocessing pipeline will later be integrated into the mobile system.
