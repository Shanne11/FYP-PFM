# Frozen Research Model Contract

`research_model_contract.json` freezes the selected experimental configuration before mobile development. The selection is based on mean federated Macro F1 and category coverage across seeds 42, 52, and 62.

The contract does **not** claim that Proposed is statistically superior to FedAvg or FedProx. It also preserves the negative finding that the current utility-weighted aggregation has negligible effect.

## Canonical checkpoint policy

Seed 42 is the canonical checkpoint because it was the predeclared default seed. It is not selected using the best held-out test score. The local artifact is expected at:

```text
outputs/class_balance/seed_42/proposed/best_global_model.pt
```

The checkpoint and `feature_pipeline.pkl` remain local because binary model and preprocessing artifacts are excluded from Git. Their hashes must be recorded during export.

## Mobile integration gates

Before Flutter inference begins:

1. Export the exact preprocessing parameters to a portable JSON representation.
2. Export the MLP to ONNX or TensorFlow Lite.
3. Verify Python and mobile inference parity on fixed non-test fixtures.
4. Decide how new merchants, accounts, payment modes, and locations map to the frozen inputs.
5. Keep utility-weighted aggregation disabled as a product claim unless a revised method is revalidated.

The fitted categorical vocabulary contains inconsistent capitalization inherited from the research dataset. Cleaning those values changes the input contract and therefore requires retraining; it must not be changed silently during mobile conversion.
