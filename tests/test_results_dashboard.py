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
    actm_selection = outputs / "actm_selection_baselines"
    actm_selection.mkdir()
    (actm_selection / "actm_selection_summary.csv").write_text(
        "strategy,prompts_mean,prompts_per_100_mean,prompt_precision_mean,"
        "wrong_to_correct_mean,correct_to_wrong_mean,net_corrections_mean,"
        "correction_yield_per_100_prompts_mean,overall_accuracy_mean,"
        "overall_macro_f1_mean\n"
        "Proposed ACTM,693,30.025997,0.848004,18.6667,18,0.6667,2.693603,"
        "0.183709,0.043183\n",
        encoding="utf-8",
    )
    threshold_analysis = outputs / "actm_threshold_analysis"
    threshold_analysis.mkdir()
    (threshold_analysis / "actm_threshold_sensitivity_summary.csv").write_text(
        "include_conflict,entropy_threshold,margin_threshold,eligibility_rate_mean,"
        "eligibility_rate_std,expected_prompt_rate_mean,budget_fill_rate_mean\n"
        "True,0.65,0.15,1.0,0.0,0.3002,1.0\n"
        "False,0.65,0.15,1.0,0.0,0.3002,1.0\n"
        "False,0.999,0.001,0.258,0.058,0.254,0.847\n",
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
    assert "How to read the results" in page
    assert "Evaluation role" in page
    assert "Plain-language reading" in page
    assert "Do not compare these scores directly with Tier A" in page
    assert "Q9: budget-matched ACTM selection comparison" in page
    assert "693" in page
    assert "No strategy dominates every outcome" in page
    assert "ACTM eligibility-threshold diagnosis" in page
    assert "25.8%" in page
