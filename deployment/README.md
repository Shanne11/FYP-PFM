# Frozen Research Model Contract

`research_model_contract.json` freezes the selected experimental configuration before mobile inference development. The selection is based on mean federated Macro F1 and category coverage across seeds 42, 52, and 62. The final Python contract ID is `pocketiq-class-weighted-proposed-seed42-v1`.

The contract does **not** claim that Proposed is statistically superior to FedAvg or FedProx. It also preserves the negative finding that the current utility-weighted aggregation has negligible effect.

## Canonical checkpoint policy

Seed 42 is the canonical checkpoint because it was the predeclared default seed. It is not selected using the best held-out test score. The local artifact is expected at:

```text
outputs/class_balance/seed_42/proposed/best_global_model.pt
```

The checkpoint and `feature_pipeline.pkl` remain local because binary model and preprocessing artifacts are excluded from Git. Their hashes must be recorded during export.

## Frozen contract contents

The generated package fixes:

- the canonical checkpoint and fitted preprocessing pipeline by SHA-256 hash;
- the 13-category output order;
- the exact ordered 105-feature input vector;
- fitted one-hot categories, scaler parameters, TF-IDF vocabulary and IDF values;
- input tensor `features` as `float32 [batch, 105]`;
- output tensor `logits` as `float32 [batch, 13]`; and
- softmax followed by category-order lookup as output postprocessing.

## Mobile integration status and remaining gates

The ONNX export, PyTorch-versus-ONNX parity gate, Flutter preprocessing implementation, ONNX Runtime integration, and 13-label-to-app-category adapter are complete. The mobile application deploys the frozen seed-42 model directly as ONNX; TensorFlow Lite conversion is not part of the implemented architecture.

Remaining release gates are:

1. Verify Python-versus-Flutter logits on the fixed non-test fixtures on a physical Android device.
2. Measure physical-device latency, memory and energy use.
3. Validate the handling of new merchants, accounts, payment modes and locations with representative statements.
4. Complete release signing, security review and device acceptance testing.
5. Keep utility-weighted aggregation disabled as a product claim unless a revised method is revalidated.

The fitted categorical vocabulary contains inconsistent capitalization inherited from the research dataset. Cleaning those values changes the input contract and therefore requires retraining; it must not be changed silently during mobile conversion.

## Reproducible ONNX export

After installing `requirements.txt`, generate the portable package from the canonical local artifacts:

```powershell
python deployment/export_mobile_package.py
```

The exporter writes `model.onnx`, exact preprocessing parameters, fixed fixture vectors, artifact hashes, and a parity report under `deployment/mobile_package/`. The ONNX binary remains local and is ignored by Git; its SHA-256 hash and the reproducibility metadata are versioned.

The canonical export passed PyTorch-versus-ONNX parity on four non-test fixtures with a maximum absolute logit difference of `0.0` at tolerance `1e-5`.
