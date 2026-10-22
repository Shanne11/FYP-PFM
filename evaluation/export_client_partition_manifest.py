"""Export the frozen client-count, category, and payment-mode composition."""

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.experiment_data import CATEGORY_ORDER, load_experiment_data


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/clean_budgetwise.csv")
    parser.add_argument("--split-manifest", default="data/experiment_split.json")
    parser.add_argument("--output", default="outputs/protocol")
    return parser.parse_args()


def main(config):
    train, _, _, manifest = load_experiment_data(config.dataset, config.split_manifest)
    work = train.copy()
    work["user_id"] = work["user_id"].fillna("Unknown").astype(str)
    work["payment_mode"] = work["payment_mode"].fillna("Unknown").astype(str)
    clients = sorted(work["user_id"].unique())
    payment_modes = sorted(work["payment_mode"].unique())
    rows = []
    for client_id in clients:
        client = work[work["user_id"] == client_id]
        row = {"client_id": client_id, "record_count": len(client)}
        category_counts = client["category"].value_counts()
        payment_counts = client["payment_mode"].value_counts()
        row.update({f"category__{label}": int(category_counts.get(label, 0)) for label in CATEGORY_ORDER})
        row.update({f"payment_mode__{label}": int(payment_counts.get(label, 0)) for label in payment_modes})
        rows.append(row)

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    composition = pd.DataFrame(rows)
    composition.to_csv(output / "client_partition_composition.csv", index=False)
    counts = composition["record_count"]
    summary = {
        "split_manifest_version": manifest["version"],
        "split_seed": manifest["seed"],
        "training_records": int(len(train)),
        "distinct_source_user_groups": int(len(clients)),
        "client_size": {
            "minimum": int(counts.min()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
            "maximum": int(counts.max()),
        },
        "category_order": list(CATEGORY_ORDER),
        "payment_modes": payment_modes,
        "source": "training partition only; deterministic source user_id grouping",
    }
    (output / "client_partition_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(clients)} client rows to {output}")


if __name__ == "__main__":
    main(arguments())
