import torch

from experiments.federated_diagnosis.run_optimisation_screen import sample_weighted_average


def test_sample_weighted_average_uses_client_record_counts():
    results = [
        {"weights": {"w": torch.tensor([1.0])}, "sample_count": 1},
        {"weights": {"w": torch.tensor([3.0])}, "sample_count": 3},
    ]
    averaged = sample_weighted_average(results)
    assert torch.allclose(averaged["w"], torch.tensor([2.5]))
