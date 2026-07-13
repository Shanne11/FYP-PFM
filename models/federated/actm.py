import numpy as np


class ACTM:
    """
    Adaptive Client Trust Mechanism

    Computes a trust score for each client based on:
    - Local accuracy
    - Local loss
    - Client dataset size
    """

    def __init__(
        self,
        alpha=0.5,
        beta=0.3,
        gamma=0.2
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_score(
        self,
        accuracy,
        loss,
        data_size
    ):
        """
        Returns a trust score.

        Higher accuracy
            -> higher trust

        Lower loss
            -> higher trust

        Larger dataset
            -> higher trust
        """

        # Avoid division by zero
        inverse_loss = 1.0 / (loss + 1e-8)

        # Dataset scaling
        data_factor = np.log1p(data_size)

        score = (

            self.alpha * accuracy +

            self.beta * inverse_loss +

            self.gamma * data_factor

        )

        return float(score)