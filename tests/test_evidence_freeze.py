from pathlib import Path

from evaluation.freeze_evidence import sha256, validate_artifact


def test_sha256_and_csv_validation(tmp_path: Path):
    artifact = tmp_path / "result.csv"
    artifact.write_text("metric,value\naccuracy,0.5\n", encoding="utf-8")

    validate_artifact(artifact)

    assert sha256(artifact) == "70f08425d80fff121000827194defc81c18c43dc860e1b12f49dde3533e084c8"
