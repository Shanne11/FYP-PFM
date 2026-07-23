"""Export frozen preprocessing and MLP to a portable ONNX package with parity evidence."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mlp import MLP


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/class_balance/seed_42/proposed/best_global_model.pt")
    parser.add_argument("--pipeline", default="outputs/class_balance/seed_42/proposed/feature_pipeline.pkl")
    parser.add_argument("--contract", default="deployment/research_model_contract.json")
    parser.add_argument("--fixtures", default="deployment/parity_fixtures.json")
    parser.add_argument("--output", default="deployment/mobile_package")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_preprocessing(pipeline):
    categorical_values = {
        column: [str(value) for value in values]
        for column, values in zip(pipeline.categorical, pipeline.categorical_encoder.categories_)
    }
    vocabulary = sorted(pipeline.note_vectorizer.vocabulary_.items(), key=lambda item: item[1])
    return {
        "schema_version": 1,
        "categorical_columns": pipeline.categorical,
        "categorical_values": categorical_values,
        "categorical_feature_order": pipeline.categorical_encoder.get_feature_names_out().tolist(),
        "numeric_columns": pipeline.numeric,
        "numeric_scaler_mean": pipeline.numeric_scaler.mean_.tolist(),
        "numeric_scaler_scale": pipeline.numeric_scaler.scale_.tolist(),
        "date_derivation": {"year": "year", "month": "month", "day": "day", "weekday": "Monday=0"},
        "note": {
            "vocabulary": {token: int(index) for token, index in vocabulary},
            "idf": pipeline.note_vectorizer.idf_.tolist(),
            "lowercase": True,
            "token_pattern": pipeline.note_vectorizer.token_pattern,
            "english_stop_words": sorted(ENGLISH_STOP_WORDS),
            "norm": pipeline.note_vectorizer.norm,
            "smooth_idf": pipeline.note_vectorizer.smooth_idf,
            "sublinear_tf": pipeline.note_vectorizer.sublinear_tf,
            "use_idf": pipeline.note_vectorizer.use_idf,
        },
        "fusion": {
            "mode": "semantic_anchor",
            "anchor_columns": ["transaction_type", "payment_mode", "location"],
            "gate": "use_note * (0.5 + 0.5 * clip(1 - cosine(note, anchor), 0, 1))",
        },
        "feature_dimensions": {
            "metadata": pipeline.metadata_size,
            "note": pipeline.note_size,
            "total": pipeline.metadata_size + pipeline.note_size,
        },
        "category_order": pipeline.category_encoder.classes_.tolist(),
    }


def fixture_features(pipeline, fixture_path):
    fixtures = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(fixtures)
    frame["category"] = pipeline.category_encoder.classes_[0]
    metadata, notes, anchors, _ = pipeline.transform_parts(frame)
    use_notes = frame["use_note"].astype(bool).to_numpy()
    matrix = pipeline.semantic_anchor_fused(metadata, notes, anchors, use_notes)
    return fixtures, matrix.toarray().astype(np.float32)


def export_onnx(model, input_size, destination):
    dummy = torch.zeros((1, input_size), dtype=torch.float32)
    torch.onnx.export(
        model, dummy, destination, input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def verify_parity(model, onnx_path, matrix, tolerance):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to run ONNX parity verification") from error
    model.eval()
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(matrix)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"features": matrix})[0]
    difference = np.abs(torch_logits - onnx_logits)
    return {
        "fixture_count": int(matrix.shape[0]), "output_values": int(difference.size),
        "max_absolute_logit_difference": float(difference.max()),
        "mean_absolute_logit_difference": float(difference.mean()),
        "tolerance": tolerance, "passed": bool(difference.max() <= tolerance),
        "torch_predicted_indices": torch_logits.argmax(axis=1).tolist(),
        "onnx_predicted_indices": onnx_logits.argmax(axis=1).tolist(),
    }


def main(config):
    output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(config.contract).read_text(encoding="utf-8"))
    pipeline = joblib.load(config.pipeline)
    preprocessing = portable_preprocessing(pipeline)
    if preprocessing["feature_dimensions"]["total"] != contract["model"]["input_features"]:
        raise ValueError("Fitted pipeline dimensions differ from frozen contract")
    if preprocessing["category_order"] != contract["categories"]:
        raise ValueError("Fitted category order differs from frozen contract")

    model = MLP(contract["model"]["input_features"], contract["model"]["output_classes"])
    model.load_state_dict(torch.load(config.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    onnx_path = output / "model.onnx"
    export_onnx(model, contract["model"]["input_features"], onnx_path)
    fixtures, matrix = fixture_features(pipeline, config.fixtures)
    parity = verify_parity(model, onnx_path, matrix, config.tolerance)
    if not parity["passed"]:
        raise ValueError(f"ONNX parity failed: {parity}")

    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    fixture_rows = []
    for fixture, vector, torch_index, onnx_index in zip(
        fixtures, matrix, parity["torch_predicted_indices"], parity["onnx_predicted_indices"]
    ):
        fixture_rows.append({
            "fixture_id": fixture["fixture_id"], "feature_vector": vector.tolist(),
            "predicted_index": torch_index, "predicted_category": contract["categories"][torch_index],
            "onnx_predicted_index": onnx_index,
        })
    (output / "parity_fixtures.json").write_text(json.dumps(fixture_rows, indent=2), encoding="utf-8")
    (output / "parity_report.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    manifest = {
        "contract_version": contract["contract_version"],
        "checkpoint_seed": contract["seed_policy"]["deployment_checkpoint_seed"],
        "onnx_opset": 17,
        "artifacts": {
            "model.onnx": sha256(onnx_path),
            "preprocessing.json": sha256(output / "preprocessing.json"),
            "source_checkpoint": sha256(config.checkpoint),
            "source_pipeline": sha256(config.pipeline),
        },
        "parity_passed": parity["passed"],
    }
    (output / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "parity": parity, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main(arguments())
