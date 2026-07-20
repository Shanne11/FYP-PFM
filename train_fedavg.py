"""Baseline 4: leakage-safe sample-weighted FedAvg."""

from utils.federated_baseline import run_federated_baseline


if __name__ == "__main__":
    run_federated_baseline("FedAvg", "outputs/baseline4", mu=0.0)
