from federated.aggregation import fedavg


class FederatedServer:

    def aggregate(

        self,

        client_weights

    ):

        return fedavg(

            client_weights

        )