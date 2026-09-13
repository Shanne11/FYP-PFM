import json
from pathlib import Path

from results_dashboard import build_page, resolve_asset


def test_resolve_asset_stays_inside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "outputs"
    allowed.mkdir()
    artifact = allowed / "result.csv"
    artifact.write_text("metric,value\naccuracy,1.0\n", encoding="utf-8")

    assert resolve_asset({"outputs": allowed}, "outputs", "result.csv") == artifact
    assert resolve_asset({"outputs": allowed}, "outputs", "../secret.txt") is None
    assert resolve_asset({"outputs": allowed}, "unknown", "result.csv") is None


def test_dashboard_renders_available_outputs(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    comparison = outputs / "comparison"
    comparison.mkdir(parents=True)
    (comparison / "baseline_comparison.csv").write_text(
        "model,accuracy,macro_f1,weighted_f1,ece,brier_score\n"
        "Proposed,0.2,0.1,0.15,0.05,0.9\n",
        encoding="utf-8",
    )
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "deployment_metrics.json").write_text(
        json.dumps({"held_out": {"accuracy": 0.9, "macro_f1": 0.8}}),
        encoding="utf-8",
    )

    page = build_page(outputs, deployment)

    assert "PocketIQ Results Dashboard" in page
    assert "Proposed" in page
    assert "0.8000" in page
    assert "Evidence boundary" in page
