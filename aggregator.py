import json
from ipfs_blockchain import get_from_ipfs, add_to_ipfs, globalSC

class WorkerResearchCenter:
    def __init__(self, name):
        self.name = name
        self.aggregated_params = None

    def aggregate_parameters(self, client_hashes):
        aggregated_model = {}
        count = 0
        for client_hash in client_hashes:
            serialized_params = get_from_ipfs(client_hash)
            local_params = json.loads(serialized_params)
            if not aggregated_model:
                aggregated_model = local_params
            else:
                for k in aggregated_model.keys():
                    aggregated_model[k] = [(a + b) / 2 for a, b in zip(aggregated_model[k], local_params[k])]
            count += 1

        if count > 0:
            self.aggregated_params = aggregated_model
            serialized_aggregated_params = json.dumps(aggregated_model)
            result = add_to_ipfs(serialized_aggregated_params)
            globalSC.update_global_model(result)
            return result
        return None
