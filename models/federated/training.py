import torch


def evaluate(

        model,

        X,

        y

):

    model.eval()

    with torch.no_grad():

        prediction = model(X)

        prediction = prediction.argmax(

            dim=1

        )

    accuracy = (

        prediction == y

    ).float().mean().item()

    return accuracy