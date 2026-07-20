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

The implementation stage is complete enough for comparative experiments. Repeated-seed evaluation, ablation studies, and complete calibration comparison remain required before the results should be treated as final Chapter 5 evidence.

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

## Latest single-seed results

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

These are single-seed results. Final reporting should include repeated seeds, means, standard deviations, ablations, ambiguous-subset results, prompting metrics, and calibration metrics.

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

1. Standardise probability-based metrics across all compatible models.
2. Add ECE and Brier score to probability-producing baselines.
3. Run at least three random seeds and report mean plus standard deviation.
4. Run the required ablations:
   - Proposed without ACTM;
   - Proposed without notes;
   - simple note concatenation;
   - without uncertainty-reduction utility;
   - without semantic-specificity utility;
   - without utility weighting;
   - alternative prompt budgets and multiplier bounds.
5. Report ambiguous-subset Macro F1 and prompt-efficiency metrics.
6. Analyse class imbalance and minority-category performance.
7. Freeze the selected model and preprocessing contract before mobile integration.

## Scope

This repository is the research and model-development environment. It is not the complete Flutter Personal Finance Management application. The selected final model, category mapping, ACTM configuration, and preprocessing pipeline will later be integrated into the mobile system.
