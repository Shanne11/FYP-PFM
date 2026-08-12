"""Freeze a reviewed trained checkpoint and preprocessing pipeline as a versioned contract."""

import argparse
import json
from pathlib import Path

import joblib


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract-id", required=True)
    parser.add_argument("--selected-method", default="merchant_aware_proposed")
    parser.add_argument("--checkpoint-seed", type=int, default=42)
    parser.add_argument("--contract-version", type=int, default=2)
    return parser.parse_args()


def main(config):
    pipeline = joblib.load(config.pipeline)
    manifest = json.loads(Path(config.split_manifest).read_text(encoding="utf-8"))
    text_vectorizer = getattr(pipeline, "transaction_text_vectorizer", None)
    if text_vectorizer is None:
        raise ValueError("Refusing to create v2 contract: pipeline has no merchant/description text block")
    categories = pipeline.category_encoder.classes_.tolist()
    contract = {
        "contract_version": config.contract_version,
        "status": "frozen_pending_mobile_parity",
        "contract_id": config.contract_id,
        "selected_method": config.selected_method,
        "selection_basis": "selected using validation evidence; test set remains final evaluation only",
        "dataset": {
            "fingerprint": manifest["dataset_fingerprint"],
            "split_manifest_version": manifest["version"],
            "train_records": len(manifest["train"]),
            "validation_records": len(manifest["validation"]),
            "test_records": len(manifest["test"]),
        },
        "seed_policy": {
            "deployment_checkpoint_seed": config.checkpoint_seed,
            "reason": "predeclared deployment seed; not selected using test performance",
        },
        "categories": categories,
        "model": {
            "type": "MLP",
            "input_features": pipeline.metadata_size + pipeline.note_size,
            "metadata_features": pipeline.metadata_size,
            "note_features": pipeline.note_size,
            "hidden_layers": [128, 64],
            "output_classes": len(categories),
            "dropout": 0.3,
            "source_checkpoint": config.checkpoint,
            "source_pipeline": config.pipeline,
            "current_format": "PyTorch state_dict",
            "tensor_contract": {
                "input_name": "features",
                "input_dtype": "float32",
                "input_shape": ["batch", pipeline.metadata_size + pipeline.note_size],
                "output_name": "logits",
                "output_dtype": "float32",
                "output_shape": ["batch", len(categories)],
                "postprocessing": "softmax logits, then argmax using categories order",
            },
        },
        "preprocessing": {
            "categorical_columns": pipeline.categorical,
            "numeric_columns": pipeline.numeric,
            "transaction_text_columns": pipeline.transaction_text_columns,
            "transaction_text_transform": "TF-IDF fitted on training partition only",
            "transaction_text_vocabulary_size": len(text_vectorizer.get_feature_names_out()),
            "note_transform": "TF-IDF fitted on training partition only",
            "note_vocabulary_size": pipeline.note_size,
            "unknown_text_handling": "tokens outside each frozen vocabulary produce zero entries",
        },
    }
    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main(arguments())
