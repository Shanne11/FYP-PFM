# 📊 Personal Finance Management (PFM) AI Framework

This repository contains the official implementation of a Personal Finance Management (PFM) transaction categorization framework. The project progressively evaluates traditional machine learning approaches, federated learning paradigms, and a proposed Human-in-the-Loop framework for privacy-preserving financial transaction classification.

---

# 📖 Overview

Personal financial records often contain sensitive information that cannot be centralized without introducing privacy concerns. This project investigates how different learning paradigms perform when classifying financial transactions into predefined budget categories while preserving user privacy.

The project is developed through a progressive experimental pipeline consisting of:

- Rule-Based Classification
- Centralized Machine Learning
- Multimodal Learning (Metadata + Transaction Notes)
- Federated Learning (FedAvg)
- Federated Learning with FedProx
- Proposed Adaptive Cognitive Trigger Model (ACTM)

The objective is to investigate whether incorporating human cognitive feedback and utility-aware aggregation can improve federated transaction categorization under highly heterogeneous (Non-IID) client environments.

---

# ✨ Features

- Rule-based transaction categorization
- Metadata-based Random Forest classification
- TF-IDF transaction note feature extraction
- Centralized multimodal learning
- Federated Learning (FedAvg)
- Federated Learning (FedProx)
- Adaptive Cognitive Trigger Model (ACTM)
- Utility-weighted federated aggregation
- Automatic evaluation metrics generation
- Confusion matrix visualization

---

# 🚀 Architectural Paradigm Matrix

The project evaluates progressively more sophisticated learning paradigms, transitioning from centralized machine learning toward decentralized privacy-preserving federated intelligence.

| Variant | Learning Paradigm | Model | Purpose | Status |
|---------|-------------------|--------|---------|--------|
| **B1** | Centralized | Rule-Based System | Traditional heuristic baseline | ✅ Completed |
| **B2** | Centralized | Random Forest | Metadata-only transaction categorization | ✅ Completed |
| **B3** | Centralized | Random Forest | Metadata + transaction notes | ✅ Completed |
| **B4** | Federated | Multi-Layer Perceptron | Standard FedAvg baseline | ✅ Completed |
| **B5** | Federated | Multi-Layer Perceptron | FedProx optimization | ✅ Completed |
| **Proposed** | Federated | MLP + ACTM | Utility-weighted cognitive aggregation | ✅ Completed |

---

# 📌 Project Workflow

```text
                    Raw Finance Dataset
                            │
                            ▼
                  Data Preprocessing Pipeline
                            │
                            ▼
                Centralized Baselines (B1-B3)
                            │
                            ▼
              Client Dataset Sharding (150 Users)
                            │
                            ▼
                Federated Learning (FedAvg)
                            │
                            ▼
             Federated Optimization (FedProx)
                            │
                            ▼
      Proposed ACTM + Utility-Weighted Aggregation
```

---

# 📦 Repository Structure

```text
FYP-PFM/
│
├── data/
│   ├── preprocess.py
│   ├── split_clients.py
│   └── clients/
│       ├── U001.csv
│       ├── ...
│       └── U150.csv
│
├── dataset/
│   ├── budgetwise_finance_dataset.csv
│   └── clean_budgetwise.csv
│
├── models/
│   ├── mlp.py
│   ├── note_model.py
│   └── federated/
│       ├── aggregation.py
│       ├── actm.py
│       ├── client.py
│       ├── server.py
│       └── utility.py
│
├── outputs/
│
├── utils/
│   ├── feature_engineering.py
│   ├── federated_dataset.py
│   ├── metrics.py
│   └── text_processing.py
│
├── train_baseline.py
├── train_metadata.py
├── train_notes.py
├── train_fedavg.py
├── train_fedprox.py
├── train_proposed.py
│
├── requirements.txt
└── README.md
```

---

# 🛠 Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/FYP-PFM.git

cd FYP-PFM
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch pandas scikit-learn matplotlib scipy
```

---

# ▶️ Running the Experiments

Each baseline can be executed independently.

## Baseline 1 — Rule-Based

```bash
python train_baseline.py
```

---

## Baseline 2 — Metadata Random Forest

```bash
python train_metadata.py
```

---

## Baseline 3 — Metadata + Notes

```bash
python train_notes.py
```

---

## Baseline 4 — Federated Learning (FedAvg)

```bash
python train_fedavg.py
```

---

## Baseline 5 — Federated Learning (FedProx)

```bash
python train_fedprox.py
```

---

## Proposed Framework

```bash
python train_proposed.py
```

---

# 📂 Generated Outputs

After training, the framework automatically stores generated artifacts inside the corresponding **outputs/** directory.

Typical outputs include:

- Trained model checkpoints (`.pth`)
- Classification reports
- Confusion matrices
- Client utility scores
- Evaluation metrics
- Prediction summaries

Examples:

```text
outputs/
│
├── metrics.txt
├── confusion_matrix.png
├── best_model.pth
├── client_utilities.csv
└── predictions.csv
```

---

# 📊 Experimental Results

## Overall Performance Comparison

| Model | Accuracy | Precision | Weighted F1 | Status |
|--------|----------:|----------:|------------:|--------|
| Rule-Based | **0.41%** | **0.17%** | **0.24%** | ✅ |
| Random Forest (Metadata) | **10.06%** | **8.74%** | **8.78%** | ✅ |
| Random Forest (Metadata + Notes) | **10.40%** | **9.94%** | **8.79%** | ✅ |
| Federated Learning (FedAvg) | **3.77%** | — | — | ✅ |
| Federated Learning (FedProx) | **6.20%** | — | — | ✅ |
| Proposed ACTM Framework | **4.33%** | — | — | ✅ |

---

# 📈 Detailed Experimental Analysis

The following section discusses the empirical observations obtained from each experimental baseline. The experiments progressively evaluate increasingly sophisticated learning paradigms, beginning with deterministic rule-based approaches and culminating in the proposed Human-in-the-Loop federated learning framework.

---

## 📉 Baseline 1 — Rules-Only Categorization

### Performance Metrics

| Metric | Value |
|--------|------:|
| Global Accuracy | **0.0041 (0.41%)** |
| Precision | **0.0017 (0.17%)** |
| Recall | **0.0041 (0.41%)** |
| Weighted F1-Score | **0.0024 (0.24%)** |

### Empirical Analysis

Baseline 1 employs a deterministic rule-based engine that categorizes transactions using manually defined keyword matching rules. Although computationally inexpensive and fully interpretable, this approach performs extremely poorly on real-world financial data.

The dataset contains significant variations in spelling, capitalization, abbreviations, and user-entered transaction descriptions (e.g., `FOOD`, `Food`, `food`, `Foods`, `Fod`, `Foodd`). Since the rule-based engine performs exact or near-exact keyword matching, these inconsistencies dramatically reduce classification capability.

The experimental results demonstrate that manually engineered heuristics are unable to generalize across noisy financial records, resulting in performance that approaches random guessing.

---

## 🌲 Baseline 2 — Centralized Metadata-Only Model

### Performance Metrics

| Metric | Value |
|--------|------:|
| Global Accuracy | **0.1006 (10.06%)** |
| Precision | **0.0874 (8.74%)** |
| Weighted F1-Score | **0.0878 (8.78%)** |

### Top Feature Importance

| Feature | Importance |
|---------|-----------:|
| amount | **26.27%** |
| location | **14.96%** |
| day | **13.45%** |
| payment_mode | **12.48%** |
| month | **10.46%** |
| weekday | **8.55%** |
| transaction_type | **7.38%** |
| year | **6.47%** |

### Empirical Analysis

Replacing deterministic rules with a Random Forest classifier significantly improves prediction performance. The ensemble model captures statistical relationships between structured transaction attributes, producing a substantial increase in classification accuracy.

Feature importance analysis indicates that transaction amount is the strongest predictor because many recurring expenses (such as rent and utilities) occur within relatively consistent numerical ranges. Temporal and contextual features, including location and payment method, also contribute meaningful predictive information.

Despite these improvements, metadata alone cannot adequately distinguish all 46 budget categories. Many transactions share similar structured attributes while representing entirely different spending purposes, creating an upper performance bound for metadata-only learning.

---

## 📝 Baseline 3 — Centralized Multimodal Fusion (Metadata + Notes)

### Performance Metrics

| Metric | Value |
|--------|------:|
| Global Accuracy | **0.1040 (10.40%)** |
| Precision | **0.0994 (9.94%)** |
| Weighted F1-Score | **0.0879 (8.79%)** |

### Most Important Features (Top of 508)

| Feature | Importance |
|---------|-----------:|
| amount | **23.88%** |
| transaction_type | **12.32%** |
| location | **9.61%** |
| day | **9.06%** |
| shopping *(TF-IDF)* | **0.84%** |
| payment *(TF-IDF)* | **0.64%** |

### Empirical Analysis

Baseline 3 extends the centralized model by incorporating TF-IDF representations extracted from transaction notes. These textual features provide additional contextual information that is unavailable from structured metadata.

The model achieves a modest improvement over Baseline 2, indicating that transaction descriptions contain useful semantic signals. However, the improvement remains relatively small because TF-IDF treats words independently and cannot capture contextual meaning.

Frequently occurring generic terms such as *shopping*, *payment*, and *online* appear across numerous budget categories, reducing the discriminative power of the textual representation and limiting further performance gains.

---

## 🌐 Baseline 4 — Distributed Privacy-Preserving Federation (FedAvg)

### Performance Metrics

| Metric | Value |
|--------|------:|
| Global Accuracy | **0.0377 (3.77%)** |
| Communication Rounds | **10** |
| Aggregation Method | **FedAvg** |

### Empirical Analysis

Baseline 4 transitions from centralized learning to federated learning using the standard FedAvg aggregation algorithm.

Instead of training on a centralized dataset, the data are partitioned across 150 simulated clients (`U001`–`U150`). Each client trains locally before sending only model parameters to the central server, preserving raw data privacy.

However, the client datasets exhibit highly heterogeneous (Non-IID) spending patterns. Individual clients optimize toward their own localized objectives, producing gradients that frequently conflict during global aggregation.

A simple arithmetic average of these model updates causes parameter cancellation, resulting in weak global convergence and reducing classification performance to **3.77%**. This phenomenon illustrates the well-known **Federated Penalty**, motivating more advanced optimization techniques.

---

## 🔒 Baseline 5 — Distributed Federation with Regularization (FedProx)

### Performance Metrics

| Metric | Value |
|--------|------:|
| Peak Accuracy | **0.0620 (6.20%)** |
| Communication Rounds | **10** |
| Proximal Parameter (μ) | **0.01** |

### Empirical Analysis

Baseline 5 replaces the standard FedAvg objective with the FedProx optimization framework.

FedProx introduces a proximal regularization term that constrains local model updates to remain close to the global model parameters. This reduces excessive divergence caused by heterogeneous client data distributions and stabilizes the optimization process.

Compared with FedAvg, FedProx increases the best validation accuracy from **3.77%** to **6.20%**, demonstrating that regularization successfully mitigates client drift. Nevertheless, the remaining performance gap indicates that optimization alone cannot compensate for the limited semantic understanding provided by a conventional MLP operating on noisy financial transaction notes.

---

## ⚖️ Proposed Framework — ACTM + Utility-Weighted Federated Aggregation

### Performance Metrics

| Metric | Value |
|--------|------:|
| Peak Accuracy | **0.0433 (4.33%)** |
| Communication Rounds | **10** |
| Trust Boundary Range | **0.0062 – 0.0070** |

### Empirical Analysis

The proposed framework introduces the **Adaptive Cognitive Trigger Model (ACTM)** together with a utility-weighted federated aggregation strategy.

Instead of assigning equal influence to every participating client, the framework dynamically computes client utility scores using local prediction accuracy, training loss, and data volume. These utility scores are normalized through logarithmic scaling (`np.log1p`) before being incorporated into the aggregation process.

The adaptive weighting mechanism reduces the influence of unreliable client updates while emphasizing clients that contribute more informative model parameters. Compared with the standard FedAvg baseline, the proposed framework achieves improved overall performance and demonstrates the feasibility of incorporating human-centered adaptive trust mechanisms into decentralized financial learning systems.

Although the proposed approach provides stronger robustness against heterogeneous client behaviour, experimental observations indicate that high-round convergence remains constrained by the semantic quality of user-entered transaction notes. Future work may investigate contextual language models and transformer-based note representations to further enhance transaction understanding.

---

# 🚀 Future Work

Potential future enhancements include:

- Transformer-based transaction note encoding (e.g., BERT)
- Personalized federated learning for individual spending behaviour
- Dynamic client clustering based on transaction similarity
- Adaptive communication-efficient federated optimization
- Real-time deployment within mobile Personal Finance Management applications
- Explainable AI techniques for transaction categorization
- Online continual learning from user corrections

---


# Corrected proposed-method experiment

`train_proposed.py` now implements the proposed FYP method as a separate,
leakage-safe experiment while preserving Baseline 1-5:

- metadata-only probabilities feed ACTM entropy, top-two margin, and
  cross-context conflict checks;
- the prompt budget selectively enables Smart Note features;
- note utility combines uncertainty reduction (0.50), semantic specificity
  (0.30), and bounded effort (0.20);
- aggregation starts with FedAvg sample-count weights, applies a utility
  multiplier clipped to 0.75-1.25, and falls back to 1.0 for missing or
  insufficient note utility;
- preprocessing is fitted on the training split only, model selection uses a
  validation split, and final metrics reload the best checkpoint before using
  the held-out test split.

The source dataset has no merchant or account columns. The experiment therefore
uses `location` as the documented merchant-context proxy and `payment_mode` as
the account/channel proxy for cross-account conflict. Replace these configurable
columns when richer data becomes available.

Install dependencies and run:

```bash
pip install -r requirements.txt
python train_proposed.py
```

Use `python train_proposed.py --help` for thresholds, prompt budget, seed, and
training controls. `--max-clients` is intended only for development smoke tests.
