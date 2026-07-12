# 📊 Personal Finance Management (PFM) AI Framework

This repository contains the implementation of centralized baselines, decentralized privacy-preserving federated learning nodes, and the proposed human-in-the-loop cognitive framework for transaction classification.

---

# 🚀 Architectural Paradigm Matrix

The project evaluates performance across a progressive benchmark layout, transitioning from traditional heuristics to decentralized, privacy-preserving client environments.

| Variant | Learning Paradigm | Model Architecture | Core Purpose / Feature Focus | Status |
|---------|-------------------|-------------------|------------------------------|--------|
| **B1** | Centralized | Rule-Based System | Traditional heuristics baseline | ✅ Completed |
| **B2** | Centralized | Random Forest | Merchant keyword & transaction metadata baseline | ✅ Completed |
| **B3** | Centralized | Random Forest | Centralized framework measuring unstructured text note contributions | ✅ Completed |
| **B4** | Federated | MLP | Standard FedAvg baseline (Privacy-preserving anchor across 150 clients) | ✅ Completed |
| **B5** | Federated | MLP | FedProx baseline (Mitigates non-IID client weight drift) | ✅ Completed |
| **Proposed** | Federated | MLP + ACTM | Utility-weighted aggregation & cognitive engagement framework | ⏳ Pending |

---

# 📌 Project Workflow

```text
Centralized Baselines (B1 - B3)
             │
             ▼
Decentralized Sharding (U001 - U150 Data Splits)
             │
             ▼
Federated Baseline: FedAvg (B4)
             │
             ▼
Explores "Federated Penalty" & Client Drift
             │
             ▼
Federated Baseline: FedProx (B5)
             │
             ▼
Mitigates Non-IID Weight Fluctuations (μ = 0.01)
             │
             ▼
Proposed Framework:
ACTM + Smart Notes + Utility-Weighted Aggregation
```

---

# 📊 Comprehensive Experimental Baseline Evaluation

## 📉 Baseline 1 — Rules-Only Categorization

### Performance Metrics

- **Global Accuracy:** **0.0041** (0.41%)
- **Precision:** **0.0017** (0.17%)
- **Recall:** **0.0041** (0.41%)
- **Weighted F1-Score:** **0.0024** (0.24%)

### Empirical Analysis

A traditional, rigid rule-based engine completely collapses under real-world data due to human entry variance. The dataset contains severe spelling inconsistencies and scattered case structures for identical financial concepts (e.g., `FOOD`, `Food`, `food`, `Foods`, `Fod`, `Foodd`). Because a hardcoded heuristic cannot calculate semantic margins, its lookup dictionary fails, yielding a near-zero performance score.

---

## 🌲 Baseline 2 — Centralized Metadata-Only Model

### Performance Metrics

- **Global Accuracy:** **0.1006** (10.06%)
- **Precision:** **0.0874** (8.74%)
- **Weighted F1-Score:** **0.0878** (8.78%)

### Feature Importance

| Feature | Importance |
|---------|-----------:|
| `amount` | **26.27%** |
| `location` | **14.96%** |
| `day` | **13.45%** |
| `payment_mode` | **12.48%** |
| `month` | **10.46%** |
| `weekday` | **8.55%** |
| `transaction_type` | **7.38%** |
| `year` | **6.47%** |

### Empirical Analysis

Transitioning from a deterministic system to an ensemble statistical classifier (Random Forest) yields an immediate performance jump. However, a metadata-only architecture hits a strict performance ceiling. This mathematically demonstrates that transactional context fields alone lack the diagnostic capacity needed to separate 46 highly granular budget labels cleanly.

---

## 📝 Baseline 3 — Centralized Multimodal Fusion (Metadata + Notes)

### Performance Metrics

- **Global Accuracy:** **0.1040** (10.40%)
- **Precision:** **0.0994** (9.94%)
- **Weighted F1-Score:** **0.0879** (8.79%)

### Most Important Features (Top of 508)

| Feature | Importance |
|---------|-----------:|
| `amount` | **23.88%** |
| `transaction_type` | **12.32%** |
| `location` | **9.61%** |
| `day` | **9.06%** |
| `shopping` *(TF-IDF)* | **0.84%** |
| `payment` *(TF-IDF)* | **0.64%** |

### Empirical Analysis

Incorporating a 500-feature sparse TF-IDF matrix from raw transaction notes provides a minor predictive lift over Baseline 2. Centralized ensemble methods struggle to extract deep semantic properties from raw text notes because high-frequency, generic tokens such as `shopping` and `payment` introduce substantial cross-category ambiguity, limiting overall performance.

---

## 🌐 Baseline 4 — Distributed Privacy-Preserving Federation (FedAvg)

### Performance Metrics

- **Global Accuracy:** **0.0377** (3.77%)
- **Communication Rounds:** **10**
- **Class Convergence:** Strongly compressed; multiple local target classes record **0.00** precision and recall due to parameter washout.

### Empirical Analysis

Moving the model from a pooled database to an isolated, multi-tenant environment reveals the **Federated Penalty**. Because the 150 clients (`U001`–`U150`) have highly unique and non-identical spending habits (Non-IID data distributions), their localized optimizations pull model parameters in opposite directions. Aggregating these updates through simple, unweighted mathematical averaging causes their directional gradients to neutralize one another. The global accuracy settles near the random guessing threshold (**1/31 ≈ 3.22%**), establishing the motivation for more advanced aggregation and regularization techniques.

---

## 🔒 Baseline 5 — Distributed Federation with Regularization (FedProx)

### Performance Metrics

- **Optimal Round Accuracy:** **0.0620** (6.20%)
- **Communication Rounds:** **10**
- **Proximal Penalty (μ):** **0.01**

### Empirical Analysis

Baseline 5 introduces proximal regularization via FedProx to reduce destructive client weight divergence observed under standard FedAvg. By incorporating a proximal penalty that constrains local optimization to remain close to the global model parameters, training becomes more stable across heterogeneous clients. This improves the peak validation accuracy from **3.77%** to **6.20%**. Nevertheless, the remaining performance gap indicates that a conventional MLP operating on noisy transaction notes still lacks sufficient semantic understanding, motivating the proposed human-in-the-loop cognitive framework.

---

# 📦 Directory Structure & Component Breakdown

```text
FYP-PFM/
├── data/
│   ├── preprocess.py                # Missing value handling, deduplication, and data sanitization
│   ├── split_clients.py             # Split clean dataset into 150 client datasets
│   └── clients/                     # U001.csv – U150.csv
│
├── dataset/
│   ├── budgetwise_finance_dataset.csv
│   └── clean_budgetwise.csv
│
├── models/
│   ├── mlp.py                       # Multi-Layer Perceptron model
│   ├── note_model.py                # Random Forest for Baseline 3
│   └── federated/
│       ├── client.py                # Local client training (FedAvg / FedProx)
│       ├── server.py                # Central server coordinator
│       └── aggregation.py           # FedAvg parameter aggregation
│
├── utils/
│   ├── feature_engineering.py       # Feature preprocessing pipeline
│   ├── text_processing.py           # TF-IDF extraction
│   ├── federated_dataset.py         # PyTorch dataset wrapper
│   └── metrics.py                   # Evaluation metrics & confusion matrices
│
├── train_baseline.py                # Baseline 1
├── train_metadata.py                # Baseline 2
├── train_notes.py                   # Baseline 3
├── train_fedavg.py                  # Baseline 4
└── train_fedprox.py                 # Baseline 5
```
