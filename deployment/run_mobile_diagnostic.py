"""Verify a Flutter diagnostic vector with the exported ONNX package.

The Flutter Model Diagnostic page copies JSON containing the exact 105-value
feature vector and Flutter probabilities. Save that JSON to a local file and
run this script to compare Python ONNX Runtime with the physical-device result.
No transaction text is uploaded or written by this script.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", nargs="?", help="Flutter diagnostic JSON file")
    parser.add_argument(
        "--stage",
        choices=("before", "after"),
        default="after",
        help="Compare the metadata-only or Smart Note feature vector",
    )
    parser.add_argument(
        "--model",
        default="deployment/mobile_package/model.onnx",
        help="Exported ONNX model",
    )
    parser.add_argument(
        "--preprocessing",
        default="deployment/mobile_package/preprocessing.json",
        help="Exported preprocessing contract",
    )
    parser.add_argument(
        "--fixture-id",
        help="Run an exported parity fixture instead of a Flutter diagnostic file",
    )
    parser.add_argument(
        "--fixtures",
        default="deployment/mobile_package/parity_fixtures.json",
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max()
    values = np.exp(shifted)
    return values / values.sum()


def load_case(config):
    if config.fixture_id:
        rows = json.loads(Path(config.fixtures).read_text(encoding="utf-8"))
        row = next(
            (item for item in rows if item.get("fixture_id") == config.fixture_id),
            None,
        )
        if row is None:
            raise ValueError(f"Unknown fixture_id: {config.fixture_id}")
        return row["feature_vector"], None, {"fixture_id": config.fixture_id}

    if not config.diagnostic:
        raise ValueError("Provide a Flutter diagnostic JSON file or --fixture-id")
    payload = json.loads(Path(config.diagnostic).read_text(encoding="utf-8"))
    suffix = "before" if config.stage == "before" else "after"
    vector = payload.get(f"feature_vector_{suffix}")
    flutter = payload.get(f"flutter_raw_{suffix}")
    if vector is None:
        raise ValueError(f"The diagnostic does not contain a {suffix} vector")
    flutter_probabilities = None if flutter is None else flutter.get("probabilities")
    return vector, flutter_probabilities, payload.get("input", {})


def run(config):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install requirements.txt before running diagnostics") from error

    preprocessing = json.loads(
        Path(config.preprocessing).read_text(encoding="utf-8")
    )
    categories = preprocessing["category_order"]
    expected_features = preprocessing["feature_dimensions"]["total"]
    vector, flutter_probabilities, case_input = load_case(config)
    if len(vector) != expected_features:
        raise ValueError(
            f"Feature length {len(vector)} does not match contract {expected_features}"
        )

    matrix = np.asarray([vector], dtype=np.float32)
    session = ort.InferenceSession(
        str(config.model), providers=["CPUExecutionProvider"]
    )
    tensor = preprocessing["tensor_contract"]
    logits = session.run(
        [tensor["output_name"]], {tensor["input_name"]: matrix}
    )[0][0]
    probabilities = softmax(logits)
    order = np.argsort(-probabilities)
    comparison = None
    passed = None
    if flutter_probabilities:
        flutter = np.asarray(flutter_probabilities, dtype=np.float64)
        if flutter.shape != probabilities.shape:
            raise ValueError("Flutter probability length differs from ONNX output")
        comparison = float(np.abs(flutter - probabilities).max())
        passed = comparison <= config.tolerance

    return {
        "stage": config.stage,
        "case": case_input,
        "feature_count": int(matrix.shape[1]),
        "python_onnx": {
            "predicted_index": int(order[0]),
            "predicted_category": categories[int(order[0])],
            "confidence": float(probabilities[order[0]]),
            "probabilities": {
                category: float(probability)
                for category, probability in zip(categories, probabilities)
            },
        },
        "flutter_comparison": {
            "available": flutter_probabilities is not None,
            "max_absolute_probability_difference": comparison,
            "tolerance": config.tolerance,
            "passed": passed,
        },
    }


def main():
    result = run(arguments())
    print(json.dumps(result, indent=2))
    if result["flutter_comparison"]["passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
