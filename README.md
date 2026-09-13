# Personal Finance Management AI Framework

This repository implements the controlled Tier A research evaluation for a clarification-aware federated transaction-categorisation framework. It compares rules, centralised machine learning, federated learning, and a proposed human-in-the-loop method based on ACTM, selective Smart Notes, an author-defined clarification-derived heuristic, and bounded heuristic-adjusted aggregation.

This research path is deliberately separate from the PocketIQ Flutter deployment model. Tier A uses the controlled BudgetWise data contract with 105 features and 13 categories; PocketIQ uses a separately trained frozen 4,096-feature, 14-category ONNX deployment package. The Tier A research checkpoint is not deployed in the mobile application.

## Current research status

The repository contains six executable methods:

| ID | Method | Learning setting | Status |
|---|---|---|---|
| B0 | Rules-only | Centralised heuristic | Implemented and rerun |
| B1 | Metadata-only Random Forest | Centralised | Implemented and rerun |
| B2 | Metadata + Notes Random Forest | Centralised | Implemented and rerun |
| B3 | FedAvg MLP | Federated | Implemented and rerun |
| B4 | FedProx MLP | Federated | Implemented and rerun |
| P | ACTM + selective Smart Notes + clarification-derived heuristic + bounded heuristic-adjusted FedAvg | Federated, human-in-the-loop | Implemented and rerun |

The corrected implementation, three-seed federated evaluation, proposed-method ablations, ACTM/prompt analysis, class-level diagnosis, class-weighted remedy and paired uncertainty analysis have been completed and rerun. ACTM currently supports budgeted prioritisation, but its frozen thresholds mark every held-out record as eligible. The results do **not** show a material benefit from the current bounded heuristic-adjusted aggregation.

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
Clarification-derived heuristic U_i
  - uncertainty reduction: 0.50
  - lexical context novelty: 0.30
  - bounded note length: 0.20
        |
        v
Local client training
        |
        v
Sample-count FedAvg contribution
  x bounded heuristic multiplier [0.75, 1.25]
        |
        v
Normalised global aggregation
```

The transaction-level score is an engineering heuristic, not a validated measure of human clarification value. Its uncertainty-change component compares different model states and may include representation, local-adaptation, and in-sample fitting effects. If a client has missing, invalid, or insufficient heuristic evidence, its multiplier returns to `1.0`. The final model is selected using validation Macro F1, reloaded from the best checkpoint, and evaluated once on the held-out test set.

Active source-code, CLI, and new outputs use `heuristic` or `signal`. Legacy `utility` aliases and duplicate CSV names remain only for reproducibility and backward compatibility; they refer to the same author-defined clarification-derived heuristic.

### Separate PocketIQ deployment package

The documented deployment trainer is implemented at [`deployment/train_deployment_model.py`](deployment/train_deployment_model.py). Download the `DoDataThings/us-bank-transaction-categories-v2` CSV, then run:

```powershell
python deployment/train_deployment_model.py `
  --dataset path/to/transactions-synthetic.csv `
  --output deployment/pocketiq_deployment_package
```

The script maps the 17 source categories to 14 PocketIQ categories, removes duplicate description/label pairs, creates a seed-42 stratified 70/15/15 split, fits the 4,096-feature word/bigram TF-IDF vocabulary on training rows only, calibrates temperature on validation rows, evaluates the held-out split, exports the linear ONNX model, runs ONNX parity checks, and writes the schema, labels, mapping, metadata, metrics, parity files, and SHA-256 manifest. It does not train or export the Tier A research MLP.

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
| Proposed | **0.1837 +/- 0.0149** | **0.0432 +/- 0.0058** | **0.0951 +/- 0.0149** | 0.0847 +/- 0.0061 | **0.9097 +/- 0.0097** |

The proposed method has the strongest mean federated accuracy, Macro F1, Weighted F1 and Brier score. Its mean ECE is effectively tied with, but marginally higher than, FedProx. All differences are small relative to the between-seed variation, and the paired analysis does not establish a statistically significant advantage.

## Proposed-method ablation results

The full method and six ablations were evaluated with the same seeds. Values are mean +/- sample standard deviation.

| Variant | Accuracy | Macro F1 | Weighted F1 | ECE | Brier score |
|---|---:|---:|---:|---:|---:|
| Full | **0.1837 +/- 0.0149** | 0.0432 +/- 0.0058 | **0.0951 +/- 0.0149** | 0.0847 +/- 0.0061 | 0.9097 +/- 0.0097 |
| Without ACTM | 0.1804 +/- 0.0096 | 0.0426 +/- 0.0060 | 0.0930 +/- 0.0108 | 0.0856 +/- 0.0062 | 0.9115 +/- 0.0064 |
| Without notes | 0.1826 +/- 0.0111 | **0.0438 +/- 0.0073** | 0.0944 +/- 0.0131 | **0.0836 +/- 0.0088** | **0.9097 +/- 0.0097** |
| Simple concatenation | 0.1830 +/- 0.0141 | 0.0430 +/- 0.0066 | 0.0951 +/- 0.0127 | 0.0839 +/- 0.0088 | 0.9097 +/- 0.0096 |
| Without uncertainty-change component | 0.1836 +/- 0.0147 | 0.0432 +/- 0.0058 | 0.0950 +/- 0.0148 | 0.0846 +/- 0.0062 | 0.9097 +/- 0.0097 |
| Without lexical-novelty component | 0.1836 +/- 0.0147 | 0.0432 +/- 0.0058 | 0.0950 +/- 0.0148 | 0.0846 +/- 0.0062 | 0.9097 +/- 0.0097 |
| Without heuristic adjustment | 0.1837 +/- 0.0149 | 0.0432 +/- 0.0058 | 0.0951 +/- 0.0149 | 0.0847 +/- 0.0061 | 0.9097 +/- 0.0097 |

### Findings supported by the ablation study

- ACTM provides the clearest positive contribution: removing it reduces mean accuracy by 0.0033, Weighted F1 by 0.0021, and worsens Brier score by 0.0018.
- Selective note use gives small accuracy and Weighted F1 gains, but the no-notes variant has slightly higher Macro F1 and better ECE. The note effect is therefore mixed.
- Semantic-anchor-gated fusion gives only a very small classification improvement over simple concatenation and does not improve calibration in these runs.
- Removing either the uncertainty-change or lexical-novelty component changes the results only negligibly.
- Removing the bounded heuristic adjustment produces effectively identical results to the full method. The current evidence does not support a claim that heuristic-adjusted aggregation materially improves performance.
- Federated Macro F1 remains low, showing that minority-category performance is still weak despite the modest aggregate improvement.

The null heuristic-adjustment result is retained as an honest research finding. The diagnostic below explains why the mechanism currently has negligible aggregation impact.

### Heuristic-adjustment diagnosis

The three full-method runs were diagnosed across 4,500 client-round observations and 57,208 selected-note observations:

| Diagnostic | Result |
|---|---:|
| Client mean heuristic | 0.3252–0.3635 |
| Heuristic multiplier | 0.9126–0.9317 |
| Multiplier standard deviation | 0.0026 |
| Within 0.01 of median multiplier | 100.0% |
| Fallback to 1.0 | 0.0% |
| Mean absolute base/final weight change | 0.0000133 |
| Mean relative base/final weight change | 0.20% |

Heuristic-component distributions show why the client means are compressed:

| Component | Mean | Standard deviation | Median | Range |
|---|---:|---:|---:|---:|
| Uncertainty reduction | 0.0056 | 0.0074 | 0.0027 | 0.0000–0.0641 |
| Lexical context novelty | 0.9199 | 0.2715 | 1.0000 | 0.0000–1.0000 |
| Bounded note-length contribution | 0.1886 | 0.0880 | 0.2500 | 0.0000–0.3750 |
| Combined transaction heuristic | 0.3165 | 0.0938 | 0.3500 | 0.0000–0.4012 |

Lexical novelty is saturated at `1.0` for most selected notes, uncertainty reduction is close to zero, and averaging many notes compresses the client-level heuristic further. Consequently, nearly uniform multipliers are removed again by final weight normalisation and have negligible aggregation impact.

**Verdict:** the current bounded heuristic-adjusted aggregation is an unsupported/negative finding. It must not be claimed as a demonstrated performance improvement. A revised mapping or novelty measure may be investigated using development data, but it must then be rerun on frozen evaluation settings and compared across seeds before any positive claim is made.

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

### Heuristic-adjustment diagnostics

After completing the three full ablation runs, audit heuristic components, client multipliers, fallback frequency, and aggregation-weight changes. The existing diagnostic executable retains its legacy `utility` name for compatibility:

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

### Class-weighted federated experiment

The first training-only remedy is inverse-frequency weighted cross-entropy. Class weights are fitted once from the frozen training partition, normalized to mean `1.0`, and reused by every client. Validation still selects the best checkpoint, and the test partition is not used to calculate or select weights.

Run the standard-versus-weighted comparison for FedAvg, FedProx, and Proposed using seeds 42, 52, and 62:

```powershell
python evaluation/run_class_balance.py
```

This performs nine new weighted runs. FedAvg and FedProx are compared with `outputs/repeated/`; Proposed is compared with the current semantic-anchor full runs in `outputs/ablations/seed_<seed>/full`. It creates:

```text
outputs/class_balance/class_balance_runs.csv
outputs/class_balance/class_balance_summary.csv
outputs/class_balance/class_balance_deltas.csv
outputs/class_balance/class_balance_macro_f1.png
```

For a non-reportable pipeline check only:

```powershell
python evaluation/run_class_balance.py --seeds 42 --rounds 1 --local-epochs 1 --max-clients 3 --output outputs/class_balance_smoke
```

Do not use the smoke-test metrics in the report. The remedy is supported only if the complete three-seed experiment improves Macro F1 and minority-class recall without an unacceptable overall trade-off.

To regenerate only the summaries from completed weighted runs without retraining:

```powershell
python evaluation/run_class_balance.py --reuse-weighted
```

#### Completed class-weighted results

The complete three-seed experiment shows a consistent class-coverage benefit with an accuracy trade-off:

| Method | Loss | Accuracy | Macro F1 | Weighted F1 | Zero-recall classes | Active predicted classes |
|---|---|---:|---:|---:|---:|---:|
| FedAvg | Standard | 0.1805 +/- 0.0094 | 0.0427 +/- 0.0065 | 0.0938 +/- 0.0092 | 10.67 | 2.67 |
| FedAvg | Class-weighted | 0.1479 +/- 0.0367 | 0.0623 +/- 0.0269 | 0.0859 +/- 0.0489 | 7.67 | 6.00 |
| FedProx | Standard | 0.1798 +/- 0.0104 | 0.0424 +/- 0.0062 | 0.0933 +/- 0.0099 | 10.67 | 2.67 |
| FedProx | Class-weighted | 0.1490 +/- 0.0354 | 0.0649 +/- 0.0252 | 0.0877 +/- 0.0474 | 7.67 | 6.33 |
| Proposed | Standard | 0.1837 +/- 0.0149 | 0.0432 +/- 0.0058 | 0.0951 +/- 0.0149 | 10.67 | 2.33 |
| Proposed | Class-weighted | 0.1502 +/- 0.0319 | **0.0661 +/- 0.0222** | 0.0872 +/- 0.0452 | 7.67 | **6.33** |

For Proposed, class weighting increases mean Macro F1 by `0.0229` (approximately 53%), increases Macro Recall by `0.0447`, reduces zero-recall categories by `3`, and increases active predicted categories by `4`. It also lowers mean accuracy by `0.0335` and Weighted F1 by `0.0080`. ECE improves by `0.0244`, while Brier score worsens slightly by `0.0027`.

This is a meaningful remedy for category collapse when Macro F1 and minority-category coverage are the primary objectives, but it is not a universal improvement across all metrics. The class-weighted Proposed method is the leading federated candidate by mean Macro F1 and category coverage; however, the completed paired analysis does not establish statistical significance. It still underperforms the centralised Metadata + Notes baseline and still leaves an average of 7.67 categories with zero recall.

#### Paired three-seed uncertainty analysis

Matched-seed comparisons quantify uncertainty around the observed Macro F1 differences:

| Comparison | Mean difference | 95% CI | Seed wins | Exact sign-flip p |
|---|---:|---:|---:|---:|
| Class-weighted vs standard FedAvg | +0.0196 | [-0.0311, 0.0703] | 2/3 | 0.50 |
| Class-weighted vs standard FedProx | +0.0225 | [-0.0248, 0.0698] | 3/3 | 0.25 |
| Class-weighted vs standard Proposed | +0.0229 | [-0.0193, 0.0651] | 3/3 | 0.25 |
| Standard Proposed vs FedAvg | +0.0005 | [-0.0033, 0.0042] | 2/3 | 0.75 |
| Standard Proposed vs FedProx | +0.0008 | [-0.0018, 0.0034] | 2/3 | 0.50 |
| Class-weighted Proposed vs FedAvg | +0.0038 | [-0.0158, 0.0234] | 2/3 | 0.75 |
| Class-weighted Proposed vs FedProx | +0.0012 | [-0.0120, 0.0145] | 1/3 | 1.00 |

With only three paired seeds, confidence intervals are wide and the smallest possible non-zero two-sided exact sign-flip p-value is `0.25`. Class weighting has a consistent positive direction for Proposed and FedProx, but no comparison establishes conventional statistical significance. The Proposed method's small mean advantage over FedAvg and FedProx is not consistent enough to support a superiority claim.

These results are exploratory effect estimates. The report may state that class weighting **shows a promising and directionally consistent Macro F1 improvement**, but it must not state that Proposed or class weighting is statistically significantly better.

### Paired statistical analysis

Generate matched-seed differences, 95% confidence intervals, paired effect sizes, win counts, and exact sign-flip p-values:

```powershell
python evaluation/paired_statistics.py
```

This creates:

```text
outputs/statistics/paired_statistics.csv
outputs/statistics/macro_f1_paired_statistics.csv
outputs/statistics/statistical_conclusion.txt
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
| Heuristic multiplier range | 0.75-1.25 |
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
- without the uncertainty-change heuristic component;
- without the lexical-novelty heuristic component;
- without bounded heuristic-adjusted aggregation.

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
- `clarification_heuristic_scores.csv` (`utility_scores.csv` compatibility copy)
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
|-- deployment/
|   `-- train_deployment_model.py
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
- sample-based aggregation weights, bounded heuristic multipliers, and fallback behaviour;
- syntax compilation and end-to-end federated smoke execution.

The latest full verification completed on 13 September 2026: all 29 Python tests passed, all reportable summary artefacts were present, 21 ablation runs, nine repeated-seed runs and nine class-balance runs were available, and the active output set contained no invalid JSON or zero-byte files. The previous output set is retained outside the repository as a recoverable archive rather than being mixed with the current evidence.

## Required next evaluation work

Before treating the results as final research evidence:

1. If the heuristic adjustment is revised, define the new mapping and novelty measure using development data only, then rerun the frozen three-seed evaluation and ablations.
2. Evaluate alternative prompt budgets and multiplier bounds without selecting settings using the final test results.
3. Calibrate ACTM thresholds on validation data if the report requires a genuinely selective ambiguous subset rather than budget-based ranking.
4. Decide whether three-seed exploratory evidence is sufficient for the FYP scope or whether additional seeds are feasible.
5. Consider further training-only remedies only if the remaining zero-recall categories are unacceptable for the final scope.
6. Keep the Tier A checkpoint and the separate PocketIQ deployment package explicitly separated in code, documentation, and reported evidence.

## Frozen Tier A research contract

The selected Tier A federated candidate is **class-weighted Proposed**, based on mean Macro F1 and category coverage rather than accuracy alone. Its exact categories, dimensions, ACTM settings, training settings, limitations, and seed policy are frozen in [`deployment/research_model_contract.json`](deployment/research_model_contract.json).

Seed 42 is the canonical Tier A reference checkpoint because it was the predeclared default—not because it achieved the best test score. Validate the contract with:

```powershell
python evaluation/validate_model_contract.py
```

This freezes the research interface for reproducibility. It does **not** identify the model deployed by PocketIQ. The current bounded heuristic adjustment remains an unsupported negative finding and must not be presented as a proven deployment benefit.

The canonical seed-42 Tier A checkpoint can be exported to ONNX for research parity checks. Four fixed non-test fixtures produced identical PyTorch and ONNX logits (`max absolute difference = 0.0`, tolerance `1e-5`). Run the reproducible export with:

```powershell
python deployment/export_mobile_package.py
```

The exported Tier A ONNX binary remains a research artefact identified by SHA-256 in `deployment/mobile_package/package_manifest.json`. It is **not** the active PocketIQ deployment model.

## Separate PocketIQ deployment path

The Tier A BudgetWise dataset does **not** contain genuine `merchant` or transaction-description fields suitable for the mobile categorisation task. The PocketIQ model is therefore trained through a separate deployment pathway using labelled transaction descriptions rather than by relabelling or deploying the Tier A checkpoint.

The deployment contract used by the [Flutter repository](https://github.com/Shanne11/FYP-personal-finance-management-system) is:

- a separate labelled transaction-description corpus;
- category mapping into 14 PocketIQ categories;
- a fixed train/validation/held-out split before data-dependent fitting;
- a training-fitted 4,096-feature word/bigram TF-IDF representation;
- runtime text composed from direction, merchant, description, and an optional Smart Note; and
- a frozen ONNX model, feature schema, category labels, metadata, mapping, manifest/checksums, and parity artefacts.

The initial corpus is the MIT-licensed synthetic [US Bank Transaction Categories v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) dataset. Its 68,000 source rows and 17 source categories are mapped and deduplicated into 45,702 records across 14 PocketIQ categories, then separated into 31,990 training, 6,856 validation and 6,856 held-out records before TF-IDF fitting.

The regenerated package reports held-out accuracy `0.9921`, Macro F1 `0.9940`, weighted F1 `0.9921`, negative log-likelihood `0.03935` and 10-bin ECE `0.00286`. Python-to-ONNX parity passed across four fixed fixtures with maximum absolute logit difference `1.65e-6`, below the predefined `1e-5` tolerance. These are synthetic US-corpus results and do not establish Malaysian real-world generalisation.

The active package is versioned as `pocketiq-deployment-text-v1` in the Flutter repository. The research and deployment paths share the clarification-aware design concept, but they have different datasets, feature contracts, category spaces, model artefacts, and evidence roles.

## Legacy optional-learning calibration harness

The localhost server in this repository was built for the legacy 13-class calibration experiment. Its 182-parameter contract is not the Tier A full-MLP federated procedure and is not compatible with the current 14-class PocketIQ deployment package without a coordinated contract update.

The reference server in `server/optional_learning_server.py`:

- binds only to `127.0.0.1`;
- requires a bearer token loaded from a private file;
- validates the frozen model contract and exact 182-parameter payload;
- rejects non-finite values and clips the submitted update norm;
- retains sample-count base weighting;
- applies a bounded 0.75-1.25 heuristic multiplier, with a 1.0 fallback when usable evidence is absent;
- stores and returns the aggregated calibration parameters; and
- never accepts raw transaction, Smart Note, merchant or account fields.

Create a private token of at least 24 characters, keep it outside version control, and run:

```powershell
python server/optional_learning_server.py --token-file .research-token --min-clients 1
```

The server is retained as a controlled localhost research harness, not a production or physical-device deployment. Do not connect the current 14-class Flutter calibration client until the server contract has been updated and parity-tested for 210 parameters.

## Scope

This repository is the controlled Python Tier A research environment. It provides algorithmic evidence for the proposed method; it is not the complete Flutter application and its 105-feature, 13-category checkpoint is not the PocketIQ deployment model. PocketIQ operationalises the user-facing clarification workflow using its own separately trained frozen deployment package.
