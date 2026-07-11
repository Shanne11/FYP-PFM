import copy

import torch


def fedavg(local_weights):

    global_weights = copy.deepcopy(

        local_weights[0]

    )

    for key in global_weights.keys():

        for i in range(1, len(local_weights)):

            global_weights[key] += local_weights[i][key]

        global_weights[key] = torch.div(

            global_weights[key],

            len(local_weights)

        )

    return global_weights