"""Freeze a reviewed Tier A checkpoint and preprocessing state as a research contract."""

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
    parser.add_argument("--selected-method", default="class_weighted_proposed")
    parser.add_argument("--checkpoint-seed", type=int, default=42)
    parser.add_argument("--contract-version", type=int, default=1)
    return parser.parse_args()


def main(config):
    pipeline = joblib.load(config.pipeline)
    manifest = json.loads(Path(config.split_manifest).read_text(encoding="utf-8"))
    categories = pipeline.category_encoder.classes_.tolist()
    input_features = pipeline.metadata_size + pipeline.note_size
    if pipeline.metadata_size != 68 or pipeline.note_size != 37 or input_features != 105:
        raise ValueError(
            "Tier A contract requires 68 metadata + 37 Smart Note = 105 features"
        )
    if len(categories) != 13:
        raise ValueError("Tier A contract requires 13 categories")
    contract = {
        "contract_version": config.contract_version,
        "status": "frozen_tier_a_research_contract",
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
            "research_seeds": [42, 52, 62],
            "research_checkpoint_seed": config.checkpoint_seed,
            "reason": "predeclared canonical research seed; not selected using test performance",
        },
        "categories": categories,
        "model": {
            "type": "MLP",
            "input_features": input_features,
            "metadata_features": pipeline.metadata_size,
            "note_features": pipeline.note_size,
            "hidden_layers": [128, 64],
            "output_classes": len(categories),
            "dropout": 0.3,
            "source_checkpoint": config.checkpoint,
            "source_pipeline": config.pipeline,
            "current_format": "PyTorch state_dict; optional ONNX research export",
            "deployment_use": False,
            "tensor_contract": {
                "input_name": "features",
                "input_dtype": "float32",
                "input_shape": ["batch", input_features],
                "output_name": "logits",
                "output_dtype": "float32",
                "output_shape": ["batch", len(categories)],
                "postprocessing": "softmax logits, then argmax using categories order",
            },
        },
        "preprocessing": {
            "categorical_columns": pipeline.categorical,
            "numeric_columns": pipeline.numeric,
            "note_transform": "TF-IDF fitted on training partition only",
            "note_vocabulary_size": pipeline.note_size,
            "unknown_note_handling": "tokens outside the frozen note vocabulary produce zero entries",
        },
    }
    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main(arguments())
