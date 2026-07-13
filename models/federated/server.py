from models.federated.utility import (
    calculate_utilities,
    utility_weighted_average
)

from models.federated.actm import ACTM


class FederatedServer:

    def __init__(self):

        self.actm = ACTM()

    def aggregate(
        self,
        client_results
    ):

        scores = []

        weights = []

        for result in client_results:

            score = self.actm.compute_score(

                accuracy=result["accuracy"],

                loss=result["loss"],

                data_size=result["data_size"]

            )

            scores.append(score)

            weights.append(
                result["weights"]
            )

        utilities = calculate_utilities(
            scores
        )

        global_weights = utility_weighted_average(
            weights,
            utilities
        )

        return global_weights, utilities