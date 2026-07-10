import torch
import torch.nn as nn

def train_local(
        model,
        X_train,
        y_train,
        epochs=5
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    for epoch in range(epochs):

        optimizer.zero_grad()

        output = model(X_train)

        loss = criterion(
            output,
            y_train
        )

        loss.backward()

        optimizer.step()


    return model.state_dict()

def fedavg(client_weights):

    global_weights = {}

    for key in client_weights[0].keys():

        global_weights[key] = sum(
            weights[key]
            for weights in client_weights
        ) / len(client_weights)


    return global_weights


class TransactionClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                num_classes
            )
        )


    def forward(self, x):

        return self.network(x)
    
    