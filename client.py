import torch
import json
from torch.optim import Adam
import torch.nn as nn
from ipfs_blockchain import add_to_ipfs, globalSC

class ClientHospital:
    def __init__(self, name, data, model):
        self.name = name
        self.data = data
        self.model = model
        self.local_params = None

    def local_training(self, epochs=5):
        optimizer = Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.data:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Client {self.name} Epoch {epoch + 1}, Loss: {running_loss / len(self.data)}")

        # Serialize and upload parameters to IPFS
        self.local_params = self.model.state_dict()
        serialized_params = json.dumps({k: v.tolist() for k, v in self.local_params.items()})
        hash_link = add_to_ipfs(serialized_params)
        return hash_link

    def share_parameters(self):
        hash_link = self.local_training()
        globalSC.update_global_model(hash_link)
        return hash_link
