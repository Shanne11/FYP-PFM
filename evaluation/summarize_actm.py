"""Summarize ACTM ambiguous-subset and prompt-efficiency evidence across seeds."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUNS = [
    "outputs/ablations/seed_42/full",
    "outputs/ablations/seed_52/full",
    "outputs/ablations/seed_62/full",
]
REPORT_METRICS = [
    "eligible_rate", "prompts_per_100", "prompt_precision", "note_acceptance_rate",
    "mean_uncertainty_reduction", "ambiguous_accuracy", "ambiguous_macro_f1",
    "ambiguous_weighted_f1", "ambiguous_ece", "ambiguous_brier_score",
]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output", default="outputs/actm_evaluation")
    return parser.parse_args()


def load_run(path):
    run = Path(path)
    prompt = pd.read_csv(run / "prompt_metrics.csv").iloc[0]
    ambiguous = pd.read_csv(run / "ambiguous_metrics.csv").iloc[0]
    overall = pd.read_csv(run / "overall_metrics.csv").iloc[0]
    row = {
        "run": f"{run.parent.name}/{run.name}",
        "transactions": int(prompt["transactions"]), "eligible": int(prompt["eligible"]),
        "eligible_rate": prompt["eligible"] / prompt["transactions"],
        "prompts": int(prompt["prompts"]), "notes_used": int(prompt["notes_used"]),
        "prompts_per_100": prompt["prompts_per_100"],
        "prompt_precision": prompt["prompt_precision"],
        "note_acceptance_rate": prompt["note_acceptance_rate"],
        "mean_uncertainty_reduction": prompt["mean_uncertainty_reduction"],
    }
    for metric in ("accuracy", "macro_f1", "weighted_f1", "ece", "brier_score"):
        row[f"ambiguous_{metric}"] = ambiguous[metric]
        row[f"ambiguous_{metric}_delta_vs_overall"] = ambiguous[metric] - overall[metric]
    return row


def summarize(runs):
    row = {"runs": len(runs)}
    for metric in REPORT_METRICS:
        row[f"{metric}_mean"] = runs[metric].mean()
        row[f"{metric}_std"] = runs[metric].std(ddof=1) if len(runs) > 1 else 0.0
    for metric in ("accuracy", "macro_f1", "weighted_f1", "ece", "brier_score"):
        delta = runs[f"ambiguous_{metric}_delta_vs_overall"]
        row[f"ambiguous_{metric}_delta_vs_overall_mean"] = delta.mean()
    return pd.DataFrame([row])


def conclusion(runs, summary):
    row = summary.iloc[0]
    full_coverage = (runs["eligible_rate"] == 1.0).all()
    lines = [
        "ACTM evaluation conclusion",
        "",
        f"Ambiguity eligibility rate: {row['eligible_rate_mean']:.2%}",
        f"Prompts per 100 transactions: {row['prompts_per_100_mean']:.2f}",
        f"Prompt precision: {row['prompt_precision_mean']:.2%} +/- {row['prompt_precision_std']:.2%}",
        f"Note acceptance rate: {row['note_acceptance_rate_mean']:.2%} +/- {row['note_acceptance_rate_std']:.2%}",
        f"Mean uncertainty reduction: {row['mean_uncertainty_reduction_mean']:.6f}",
        f"Ambiguous Macro F1: {row['ambiguous_macro_f1_mean']:.4f} +/- {row['ambiguous_macro_f1_std']:.4f}",
        "",
    ]
    if full_coverage:
        lines.extend([
            "All held-out transactions were eligible under at least one ambiguity condition.",
            "Therefore the reported ambiguous-subset metrics equal the overall test metrics and",
            "do not demonstrate discrimination of a smaller ambiguous subset. The 30% prompt",
            "budget, rather than the thresholds alone, controls selectivity in these runs.",
        ])
    lines.extend([
        "Prompt precision and acceptance are high, but uncertainty reduction is negligible.",
        "Report ACTM as useful for budgeted prioritisation, not as a validated uncertainty-reduction mechanism.",
    ])
    return "\n".join(lines) + "\n"


def main(config):
    output = Path(config.output); output.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([load_run(path) for path in config.runs])
    summary = summarize(runs)
    runs.to_csv(output / "actm_runs.csv", index=False)
    summary.to_csv(output / "actm_summary.csv", index=False)
    text = conclusion(runs, summary)
    (output / "actm_conclusion.txt").write_text(text, encoding="utf-8")

    chart = summary.iloc[0][[
        "eligible_rate_mean", "prompt_precision_mean", "note_acceptance_rate_mean",
    ]].rename({
        "eligible_rate_mean": "Eligible", "prompt_precision_mean": "Prompt precision",
        "note_acceptance_rate_mean": "Note acceptance",
    })
    plt.figure(figsize=(7, 5)); chart.plot(kind="bar", ylim=(0, 1.05))
    plt.ylabel("Mean rate across seeds"); plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); plt.savefig(output / "actm_rates.png", dpi=160); plt.close()
    print(summary.to_string(index=False)); print("\n" + text)


if __name__ == "__main__":
    main(arguments())
