import numpy as np
import copy
import torch

def calculate_utilities(scores):
    """
    Convert ACTM trust scores
    into normalized aggregation weights.

    Sum(weights) = 1
    """

    scores = np.array(scores)

    scores = np.maximum(scores, 0)

    if scores.sum() == 0:

        return np.ones(len(scores)) / len(scores)

    utilities = scores / scores.sum()

    return utilities

def utility_weighted_average(
    client_weights,
    utilities
):
    """
    Utility-weighted FedAvg

    Instead of

        1/N

    use

        utility_i
    """

    global_weights = copy.deepcopy(
        client_weights[0]
    )

    for key in global_weights.keys():

        global_weights[key] *= utilities[0]

        for i in range(1, len(client_weights)):

            global_weights[key] += (

                client_weights[i][key]

                * utilities[i]

            )

    return global_weights