"""Baseline 5: FedAvg-equivalent setup with only the FedProx term added."""

from utils.federated_baseline import run_federated_baseline


if __name__ == "__main__":
    run_federated_baseline("FedProx", "outputs/baseline5", mu=0.01)
