# 📊 Personal Finance Management (PFM) AI Framework

This repository contains the implementation of centralized baselines, decentralized privacy-preserving federated learning nodes, and the proposed human-in-the-loop cognitive framework for transaction classification.

---

## 🚀 Architectural Paradigm Matrix

The project evaluates performance across a progressive benchmark layout, transitioning from traditional heuristics to decentralized, privacy-preserving client environments.

| Variant | Learning Paradigm | Model Architecture | Core Purpose / Feature Focus | Status |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | Centralized | Rule-Based System | Traditional heuristics baseline | ✅ Completed |
| **B2** | Centralized | Random Forest | Merchant keyword & transaction metadata baseline | ✅ Completed |
| **B3** | Centralized | Random Forest | Centralized framework measuring unstructured text note contributions | ✅ Completed |
| **B4** | Federated | MLP | Standard FedAvg baseline (Privacy-preserving anchor across 150 clients) | ✅ Completed |
| **B5** | Federated | MLP | FedProx baseline (Mitigates non-IID client weight drift) | 🚧 In Progress |
| **Proposed**| Federated | MLP + ACTM | Utility-weighted aggregation & cognitive engagement framework | ⏳ Pending |

---

## 📌 Project Workflow

```text
Centralized Baselines (B1 - B3)
             │
             ▼
Decentralized Sharding (U001 - U150 Data Splits)
             │
             ▼
Federated Baseline: FedAvg (B4) ───► Explores "Federated Penalty" & Client Drift
             │
             ▼
Federated Baseline: FedProx (B5) ───► Mitigates Non-IID Weight Fluctuations
             │
             ▼
Proposed Framework: ACTM + Smart Notes + Utility-Weighted Aggregation
