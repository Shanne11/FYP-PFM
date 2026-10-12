"""Create the provisional Section 16 evidence ZIP with declared artefacts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLUTTER_ROOT = ROOT.parent / "FYP-personal-finance-management-system"
AUDIT = ROOT / "outputs" / "section16_audit"
OUTPUT = AUDIT / "Section16_Evidence_Package_PROVISIONAL_2026-10-11.zip"
SUPERVISOR_PDF = Path(r"D:\thepr\Download\第10组_Shannell Wong Si Yuan_Clarification-Aware Federated Transaction Categorisation (1).pdf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    evidence_manifest_path = ROOT / "outputs" / "evidence_freeze" / "evidence_manifest.json"
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
    package_entries: list[dict[str, object]] = []

    def add_file(archive: zipfile.ZipFile, source: Path, destination: str) -> None:
        if not source.exists():
            raise FileNotFoundError(source)
        archive.write(source, destination)
        package_entries.append({
            "archive_path": destination,
            "source_path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        })

    readme = """# Section 16 Evidence Package — Provisional

Purpose: link every Section 16 numerical result and implementation claim to an experiment output, source implementation or test record.

Status: provisional because the Python and Flutter working trees contain uncommitted changes. The 26 declared artefacts match the existing SHA-256 manifest, but a final freeze must be regenerated after both repositories are committed.

Critical corrections:
- Q12: the supplied final-wrong formula reproduces 874 cases; the stricter implemented wrong-before-and-after metric reproduces 845. Neither supports 900.
- Q15: matched final-test client accuracy reproduces 0 improved, 0 declined and 450 unchanged client-seed cases, not 15/19/416.

Evidence boundaries:
- BudgetWise notes are retrospective source-note proxies, not participant clarification responses.
- PocketIQ Flutter uses a separate 4,096-feature, 14-category ONNX contract.
- The Tier A demonstration trace is not a PocketIQ on-device audit record.
- Q18 passes source inspection and a direct programmatic sync rejection test, but the Flutter changes still require a final commit.

Open `audit/Section16_Evidence_Matrix.xlsx` for the main review view.
"""

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("PACKAGE_README.md", readme)
        add_file(archive, evidence_manifest_path, "manifest/evidence_manifest.json")
        for artifact in evidence_manifest["artifacts"]:
            source = ROOT / artifact["path"]
            add_file(archive, source, f"declared_artifacts/{artifact['path']}")

        for source in sorted(AUDIT.iterdir()):
            if source.is_file() and source != OUTPUT and not source.name.endswith(".zip"):
                add_file(archive, source, f"audit/{source.name}")

        python_sources = [
            ROOT / "evaluation" / "audit_section16_evidence.py",
            ROOT / "evaluation" / "build_section16_workbook.py",
            ROOT / "evaluation" / "run_actm_selection_baselines.py",
            ROOT / "models" / "federated" / "actm.py",
            ROOT / "models" / "federated" / "heuristic.py",
            ROOT / "utils" / "proposed_features.py",
            ROOT / "train_proposed.py",
        ]
        for source in python_sources:
            add_file(archive, source, f"source/python/{source.relative_to(ROOT).as_posix()}")

        flutter_sources = [
            FLUTTER_ROOT / "lib" / "ml" / "learning" / "optional_learning_service.dart",
            FLUTTER_ROOT / "test" / "optional_learning_upload_disabled_test.dart",
            FLUTTER_ROOT / "lib" / "ml" / "inference" / "onnx_transaction_categoriser.dart",
            FLUTTER_ROOT / "lib" / "ml" / "inference" / "feature_preprocessor.dart",
        ]
        for source in flutter_sources:
            add_file(archive, source, f"source/flutter/{source.relative_to(FLUTTER_ROOT).as_posix()}")

        if SUPERVISOR_PDF.exists():
            add_file(archive, SUPERVISOR_PDF, "source_context/supervisor_feedback.pdf")

        package_manifest = {
            "package_status": "provisional-uncommitted-changes",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "declared_artifact_count": evidence_manifest["artifact_count"],
            "declared_manifest_head": evidence_manifest["repository_head"],
            "entries": package_entries,
        }
        archive.writestr(
            "manifest/package_contents.json",
            json.dumps(package_manifest, indent=2, ensure_ascii=False),
        )

    digest = sha256(OUTPUT)
    hash_path = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256.txt")
    hash_path.write_text(f"{digest}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps({
        "zip": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": digest,
        "declared_artifacts": evidence_manifest["artifact_count"],
        "packaged_files": len(package_entries) + 2,
    }, indent=2))


if __name__ == "__main__":
    main()
