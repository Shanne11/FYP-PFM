"""Train and export PocketIQ's separate 14-category ONNX deployment model.

This pipeline is independent of the BudgetWise Tier A research checkpoint.
It expects the US Bank Transaction Categories v2 CSV with ``description`` and
``category`` columns, maps the 17 source labels to PocketIQ's 14 labels, fits a
4,096-feature word/bigram TF-IDF representation on training rows only, selects
temperature calibration on validation rows, and evaluates held-out rows only
after the package is fixed.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split


SOURCE = "DoDataThings/us-bank-transaction-categories-v2"
CATEGORY_MAPPING = {
    "Education": "Education",
    "Entertainment": "Entertainment",
    "Fees": "Fees & Charges",
    "Groceries": "Shopping",
    "Healthcare": "Health & Medical",
    "Income": "Other Income",
    "Insurance": "Insurance",
    "Mortgage": "Housing",
    "Personal Care": "Personal Care",
    "Rent": "Housing",
    "Restaurants": "Food & Dining",
    "Shopping": "Shopping",
    "Subscription": "Bills & Utilities",
    "Transfer": "Transfers",
    "Transportation": "Transportation",
    "Travel": "Travel",
    "Utilities": "Bills & Utilities",
}
CATEGORY_ORDER = sorted(set(CATEGORY_MAPPING.values()))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Downloaded source CSV.")
    parser.add_argument("--output", default="deployment/pocketiq_deployment_package")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def prepare_dataset(path):
    frame = pd.read_csv(path)
    required = {"description", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"dataset must contain columns: {sorted(required)}")
    work = frame[["description", "category"]].copy()
    work["description"] = work["description"].fillna("").astype(str).str.strip()
    work["source_category"] = work["category"].fillna("").astype(str).str.strip()
    unknown = sorted(set(work["source_category"]) - set(CATEGORY_MAPPING))
    if unknown:
        raise ValueError(f"unmapped source categories: {unknown}")
    work["category"] = work["source_category"].map(CATEGORY_MAPPING)
    work = work[work["description"].ne("")]
    work = work.drop_duplicates(subset=["description", "category"], keep="first")
    return work.reset_index(drop=True)


def fixed_split(frame, seed):
    train, remainder = train_test_split(
        frame, test_size=0.30, random_state=seed, stratify=frame["category"]
    )
    validation, held_out = train_test_split(
        remainder, test_size=0.50, random_state=seed,
        stratify=remainder["category"],
    )
    return tuple(part.reset_index(drop=True) for part in (train, validation, held_out))


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def expected_calibration_error(probabilities, labels, bins=10):
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(
                (predicted[selected] == labels[selected]).mean()
                - confidence[selected].mean()
            )
    return float(value)


def select_temperature(logits, labels):
    candidates = np.linspace(0.5, 2.0, 61)
    losses = [log_loss(labels, softmax(logits / value), labels=np.arange(len(CATEGORY_ORDER)))
              for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def metric_summary(probabilities, labels):
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted")),
        "negative_log_likelihood": float(
            log_loss(labels, probabilities, labels=np.arange(len(CATEGORY_ORDER)))
        ),
        "ece_10_equal_width": expected_calibration_error(probabilities, labels),
    }


def export_linear_onnx(classifier, temperature, feature_count, output_path):
    weights = (classifier.coef_.T / temperature).astype(np.float32)
    bias = (classifier.intercept_ / temperature).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weights"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["logits"]),
        ],
        "pocketiq_deployment_text",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, feature_count])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, len(CATEGORY_ORDER)])],
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(
        graph,
        producer_name="PocketIQ deployment training pipeline",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parity(vectorizer, classifier, temperature, model_path, tolerance):
    fixtures = [
        ("restaurant_without_note", "debit Starbucks STARBUCKS #1234"),
        ("transfer_without_note", "debit DuitNow WIRE TRANSFER TO NAME"),
        ("salary_without_note", "credit Employer ACME CORP PAYROLL"),
        ("ambiguous_with_note", "debit Grab GRAB PAYMENT food delivery"),
    ]
    features = vectorizer.transform([text for _, text in fixtures]).astype(np.float32)
    expected = classifier.decision_function(features) / temperature
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"features": features.toarray()})[0]
    differences = np.abs(expected - actual)
    rows = []
    probabilities = softmax(actual)
    for index, (fixture_id, text) in enumerate(fixtures):
        rows.append({
            "fixture_id": fixture_id,
            "composed_text": text,
            "feature_vector": features[index].toarray().ravel().tolist(),
            "predicted_index": int(probabilities[index].argmax()),
            "predicted_category": CATEGORY_ORDER[int(probabilities[index].argmax())],
            "confidence": float(probabilities[index].max()),
        })
    report = {
        "rows": len(rows),
        "max_absolute_logit_difference": float(differences.max()),
        "mean_absolute_logit_difference": float(differences.mean()),
        "tolerance": tolerance,
        "passed": bool(differences.max() <= tolerance),
    }
    return rows, report


def main(config):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_dataset(config.dataset)
    train, validation, held_out = fixed_split(data, config.seed)
    vectorizer = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 2),
        max_features=config.max_features, sublinear_tf=True, norm="l2",
    )
    train_x = vectorizer.fit_transform(train["description"])
    if train_x.shape[1] != config.max_features:
        raise ValueError(
            f"training corpus produced {train_x.shape[1]} features; expected {config.max_features}"
        )
    validation_x = vectorizer.transform(validation["description"])
    held_out_x = vectorizer.transform(held_out["description"])
    label_to_index = {label: index for index, label in enumerate(CATEGORY_ORDER)}
    train_y = train["category"].map(label_to_index).to_numpy()
    validation_y = validation["category"].map(label_to_index).to_numpy()
    held_out_y = held_out["category"].map(label_to_index).to_numpy()

    classifier = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-5, max_iter=2000, tol=1e-4,
        class_weight="balanced", random_state=config.seed,
    ).fit(train_x, train_y)
    if classifier.classes_.tolist() != list(range(len(CATEGORY_ORDER))):
        raise ValueError("classifier class order does not match category order")
    validation_logits = classifier.decision_function(validation_x)
    temperature = select_temperature(validation_logits, validation_y)
    validation_probabilities = softmax(validation_logits / temperature)
    held_out_probabilities = softmax(classifier.decision_function(held_out_x) / temperature)
    model_path = output / "model.onnx"
    export_linear_onnx(classifier, temperature, config.max_features, model_path)

    feature_names = vectorizer.get_feature_names_out().tolist()
    schema = {
        "schema_version": 2,
        "model_family": "pocketiq_deployment_text",
        "category_space": "pocketiq",
        "text": {
            "columns": ["direction", "merchant", "description", "smart_note"],
            "composition": "direction merchant description smart_note",
            "lowercase": True,
            "token_pattern": r"(?u)\b\w\w+\b",
            "ngram_range": [1, 2],
            "norm": "l2",
            "sublinear_tf": True,
            "english_stop_words": [],
            "fit_scope": "training split only",
            "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
            "idf": vectorizer.idf_.tolist(),
        },
        "feature_dimensions": {"text": config.max_features, "total": config.max_features},
        "feature_order": [f"deployment_tfidf__{name}" for name in feature_names],
        "category_order": CATEGORY_ORDER,
        "inference_policy": {
            "confidence_threshold": config.confidence_threshold,
            "validation_accepted_accuracy": float(
                (validation_probabilities.argmax(axis=1)[
                    validation_probabilities.max(axis=1) >= config.confidence_threshold
                ] == validation_y[
                    validation_probabilities.max(axis=1) >= config.confidence_threshold
                ]).mean()
            ),
            "validation_coverage": float(
                (validation_probabilities.max(axis=1) >= config.confidence_threshold).mean()
            ),
            "temperature": temperature,
            "low_confidence_action": "ACTM clarification and provisional direction-safe fallback",
        },
        "tensor_contract": {
            "input_name": "features", "input_dtype": "float32",
            "input_shape": ["batch", config.max_features],
            "output_name": "logits", "output_dtype": "float32",
            "output_shape": ["batch", len(CATEGORY_ORDER)],
            "postprocessing": "softmax calibrated logits, then argmax using category_order",
        },
    }
    write_json(output / "feature_schema.json", schema)
    write_json(output / "category_labels.json", CATEGORY_ORDER)
    with (output / "category_mapping.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_category", "pocketiq_category"])
        writer.writerows(CATEGORY_MAPPING.items())
    write_json(output / "model_metadata.json", {
        "runtimeStatus": "trained_deployment_inference",
        "modelFamily": "description_based_linear_tfidf",
        "contractStatus": "deployment_candidate_v1",
        "contractId": "pocketiq-deployment-text-v1",
        "contractVersion": 1,
        "modelVersion": "pocketiq-deployment-text-v1",
        "runtimeFormat": "ONNX Runtime",
        "inputShape": [1, config.max_features], "inputName": "features",
        "inputDtype": "float32", "outputClasses": len(CATEGORY_ORDER),
        "outputShape": [1, len(CATEGORY_ORDER)], "outputName": "logits",
        "outputDtype": "float32",
        "postprocessing": "softmax calibrated logits, then argmax using category_order",
        "confidenceThreshold": config.confidence_threshold,
        "entropyThreshold": 0.65, "marginThreshold": 0.15,
        "isProductionEnabled": True,
        "researchBoundary": "Separate from the BudgetWise Tier A research checkpoint",
    })
    accepted = validation_probabilities.max(axis=1) >= config.confidence_threshold
    metrics = {
        "dataset": {
            "source": SOURCE, "licence": "MIT", "synthetic": True,
            "rows_after_deduplication": len(data), "train_rows": len(train),
            "validation_rows": len(validation), "held_out_rows": len(held_out),
            "source_categories": sorted(CATEGORY_MAPPING),
            "pocketiq_categories": CATEGORY_ORDER,
        },
        "validation": metric_summary(validation_probabilities, validation_y),
        "held_out": metric_summary(held_out_probabilities, held_out_y),
        "confidence_policy": {
            "threshold": config.confidence_threshold,
            "coverage": float(accepted.mean()),
            "accepted_accuracy": float(
                (validation_probabilities.argmax(axis=1)[accepted] == validation_y[accepted]).mean()
            ),
        },
        "temperature": temperature,
        "boundary": "Synthetic US descriptions initialise the deployment model; independent Malaysian held-out evaluation is required before real-world claims.",
    }
    write_json(output / "deployment_metrics.json", metrics)
    fixtures, parity = build_parity(
        vectorizer, classifier, temperature, model_path, config.parity_tolerance
    )
    write_json(output / "parity_fixtures.json", fixtures)
    write_json(output / "parity_report.json", parity)
    if not parity["passed"]:
        raise RuntimeError(f"ONNX parity failed: {parity}")
    artifact_names = [
        "model.onnx", "feature_schema.json", "category_labels.json",
        "model_metadata.json", "deployment_metrics.json", "parity_fixtures.json",
        "parity_report.json", "category_mapping.csv",
    ]
    write_json(output / "package_manifest.json", {
        "package": "pocketiq-deployment-text-v1",
        "created_by": "deployment/train_deployment_model.py",
        "artifacts": {name: sha256(output / name) for name in artifact_names},
        "parity_passed": True,
    })
    print(json.dumps({"output": str(output), "metrics": metrics, "parity": parity}, indent=2))


if __name__ == "__main__":
    main(arguments())
