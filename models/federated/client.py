import copy
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class FederatedClient:

    def __init__(
        self,
        model,
        learning_rate=0.001
    ):

        self.model = model

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

    def train(
        self,
        X,
        y,
        epochs=5,
        batch_size=32,
        global_weights=None,
        mu=0
    ):

        dataset = TensorDataset(X, y)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.model.train()

        total_loss = 0

        correct = 0

        total = 0

        for _ in range(epochs):

            for batch_x, batch_y in loader:

                self.optimizer.zero_grad()

                output = self.model(batch_x)

                loss = self.criterion(
                    output,
                    batch_y
                )

                # FedProx proximal term
                if global_weights is not None:

                    proximal_term = 0

                    for name, param in self.model.named_parameters():

                        proximal_term += (
                            (param - global_weights[name]) ** 2
                        ).sum()

                    loss += (mu / 2) * proximal_term

                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

                prediction = torch.argmax(
                    output,
                    dim=1
                )

                correct += (
                    prediction == batch_y
                ).sum().item()

                total += len(batch_y)

        accuracy = correct / total

        average_loss = total_loss / len(loader)

        return {

            "weights": copy.deepcopy(
                self.model.state_dict()
            ),

            "accuracy": accuracy,

            "loss": average_loss,

            "data_size": total

        }