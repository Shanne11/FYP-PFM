import copy

import torch

from experiments.federated_diagnosis.run_adam_persistence import PersistentAdamClient
from models.mlp import MLP


def _tiny_batch():
    torch.manual_seed(7)
    return torch.randn(8, 3), torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])


def test_persistent_adam_step_continues_after_global_parameter_reload():
    torch.manual_seed(3)
    global_model = MLP(3, 2)
    client = PersistentAdamClient(copy.deepcopy(global_model), learning_rate=0.001)
    features, targets = _tiny_batch()
    client.train(features, targets, epochs=1)
    first_step = client.optimiser_step()
    client.load_global_parameters(global_model.state_dict())
    client.train(features, targets, epochs=1)
    assert first_step > 0
    assert client.optimiser_step() > first_step


def test_persistent_adam_states_are_isolated_between_clients():
    torch.manual_seed(3)
    global_model = MLP(3, 2)
    client_a = PersistentAdamClient(copy.deepcopy(global_model), learning_rate=0.001)
    client_b = PersistentAdamClient(copy.deepcopy(global_model), learning_rate=0.001)
    features, targets = _tiny_batch()
    client_a.train(features, targets, epochs=1)
    assert client_a.optimiser is not client_b.optimiser
    assert client_a.optimiser_step() > 0
    assert client_b.optimiser_step() == 0
    assert all(a is not b for a, b in zip(client_a.model.parameters(), client_b.model.parameters()))
