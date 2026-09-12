# Research Export and PocketIQ Deployment Training

This directory contains two separate model paths. They share the clarification-aware research concept, but they do not share a dataset, feature contract, category space, or checkpoint.

## Tier A research export

`research_model_contract.json` freezes the selected controlled Tier A configuration. It uses the BudgetWise research data, 105 features, 13 categories, and the full MLP evaluated across 150 simulated clients. The selected checkpoint is based on validation Macro F1 and category coverage across seeds 42, 52, and 62. The contract does not establish superiority over FedAvg or FedProx, and the current bounded heuristic adjustment remains an unsupported/negative experimental finding.

`export_mobile_package.py` is retained to reproduce and verify a portable ONNX form of this research checkpoint. Its output under `deployment/mobile_package/` is a research artefact only. It is **not** the model loaded by PocketIQ Flutter.

## PocketIQ deployment model

`train_deployment_model.py` trains the separate PocketIQ deployment model from a downloaded US Bank Transaction Categories v2 CSV. The pipeline:

- requires labelled transaction descriptions;
- maps the 17 source categories to 14 PocketIQ categories;
- removes duplicate description-and-category pairs;
- creates a fixed seed-42 70/15/15 train, validation, and held-out split;
- fits a lowercase 4,096-feature word-and-bigram TF-IDF representation on training descriptions only;
- trains a linear classifier and selects temperature calibration on validation data;
- evaluates the held-out deployment partition after the contract is fixed; and
- exports the ONNX model, feature schema, category labels, mapping, metadata, parity fixtures, parity report, and SHA-256 manifest.

Run it after installing `requirements.txt`:

```powershell
python deployment/train_deployment_model.py --dataset path\to\transactions.csv
```

The default output directory is `deployment/pocketiq_deployment_package/`. Copy a package into Flutter only after its Python-to-ONNX parity check passes and its 14-category schema matches the application assets.

At runtime, Flutter composes transaction direction, merchant, description, and an optional Smart Note using the frozen deployment text contract. Transparent local rules run first; unresolved records use the frozen 14-category ONNX model. PocketIQ ACTM Review operates per transaction. When the user submits a non-empty Smart Note, the application reruns the same frozen ONNX model and then checks the transparent Smart Note rule before presenting the recommendation for confirmation or correction.

## Evidence boundary

The deployment corpus is synthetic and US-oriented. It supports deployment preparation, parity, and functional verification, not Malaysian real-world accuracy claims. A separately labelled Malaysian held-out set is required for those claims. Full federated training and bounded signal-adjusted aggregation are evaluated only in Python Tier A and are not executed by Flutter.
