"""Baseline 4: leakage-safe sample-weighted FedAvg."""

import argparse

from utils.federated_baseline import run_federated_baseline
from utils.text_feature_options import add_transaction_text_arguments


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/clean_budgetwise.csv")
    parser.add_argument("--split-manifest", default="data/experiment_split.json")
    parser.add_argument("--output", default="outputs/baseline4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--class-weighted-loss", action="store_true")
    add_transaction_text_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    config = arguments()
    run_federated_baseline(
        "FedAvg", config.output, mu=0.0, dataset=config.dataset,
        split_manifest=config.split_manifest,
        seed=config.seed, rounds=config.rounds, local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
        class_weighted_loss=config.class_weighted_loss,
        transaction_text_columns=config.transaction_text_columns,
        max_transaction_text_features=config.max_transaction_text_features,
    )
