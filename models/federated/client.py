import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn as nn


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
            batch_size=32
    ):

        dataset = TensorDataset(X, y)

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

                loss.backward()

                self.optimizer.step()

        return self.model.state_dict()