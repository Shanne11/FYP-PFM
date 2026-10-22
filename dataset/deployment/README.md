# PocketIQ deployment-training data

`transactions-synthetic.csv` is the MIT-licensed synthetic dataset published as
`DoDataThings/us-bank-transaction-categories-v2`. It is used only to initialise
the separate PocketIQ text deployment model.

This file does not replace the BudgetWise Tier A research dataset or any Tier A
baseline/proposed-model result. The source descriptions are US-oriented and
synthetic, so the exported model is a deployment candidate rather than evidence
of Malaysian real-world accuracy. A separately labelled Malaysian held-out set
is still required for that claim.

Rebuild the deployment package from the repository root with:

```powershell
python deployment/train_deployment_model.py `
  --dataset dataset/deployment/transactions-synthetic.csv `
  --output deployment/pocketiq_deployment_package
```

The script fits preprocessing only on the training split, calibrates temperature
and the confidence policy on validation data, evaluates once on the held-out
split, verifies ONNX parity, and emits the immutable Flutter package artifacts.
