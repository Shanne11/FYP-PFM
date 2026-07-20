"""Build the final like-for-like comparison after all six methods are run."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODELS = {
    "Rules-only": "baseline1",
    "Metadata-only": "baseline2",
    "Metadata + Notes": "baseline3",
    "FedAvg": "baseline4",
    "FedProx": "baseline5",
    "Proposed": "proposed",
}


def main():
    rows = []
    for model, folder in MODELS.items():
        path = Path("outputs") / folder / "overall_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run {model} first; missing {path}")
        row = pd.read_csv(path).iloc[0].to_dict(); row["model"] = model; rows.append(row)
    comparison = pd.DataFrame(rows)
    columns = ["model"] + [column for column in comparison if column != "model"]
    comparison = comparison[columns]
    output = Path("outputs/comparison"); output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "baseline_comparison.csv", index=False)
    chart = comparison.set_index("model")[["accuracy", "macro_f1"]]
    chart.plot(kind="bar", figsize=(10, 6)); plt.ylabel("Score"); plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(output / "baseline_comparison.png", dpi=160); plt.close()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
