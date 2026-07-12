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

            mu=0.0

    ):

        dataset = TensorDataset(
            X,
            y
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.model.train()

        for _ in range(epochs):

            for batch_x, batch_y in loader:

                self.optimizer.zero_grad()

                output = self.model(batch_x)

                loss = self.criterion(
                    output,
                    batch_y
                )

                # -----------------------------
                # FedProx proximal term
                # -----------------------------

                if global_weights is not None and mu > 0:

                    proximal_loss = 0.0

                    for name, param in self.model.named_parameters():

                        proximal_loss += torch.norm(

                            param -

                            global_weights[name]

                        ) ** 2

                    loss += (mu / 2.0) * proximal_loss

                loss.backward()

                self.optimizer.step()

        return copy.deepcopy(
            self.model.state_dict()
        )