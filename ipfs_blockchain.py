import requests
import json

IPFS_API_URL = "http://localhost:5001/api/v0"

def add_to_ipfs(data):
    res = requests.post(f"{IPFS_API_URL}/add", files={"file": data.encode('utf-8')})
    return res.json()["Hash"]

def get_from_ipfs(hash_link):
    res = requests.post(f"{IPFS_API_URL}/cat?arg={hash_link}")
    return res.text

class SmartContract:
    def __init__(self):
        self.participants = []
        self.global_model_hash = ""
        self.participant_rewards = {}

    def register_participant(self, participant):
        self.participants.append(participant)
        print(f"Participant {participant} registered.")

    def update_global_model(self, hash_link):
        self.global_model_hash = hash_link
        print(f"Global model updated with hash: {hash_link}")

    def reward_participant(self, participant, amount):
        if participant in self.participant_rewards:
            self.participant_rewards[participant] += amount
        else:
            self.participant_rewards[participant] = amount
        print(f"Participant {participant} rewarded with {amount} tokens.")

    def get_global_model_hash(self):
        return self.global_model_hash

globalSC = SmartContract()
rewardSC = SmartContract()
