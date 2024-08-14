import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from config import config
from dataset import BratsDataset
from models import SimpleModel
from client import ClientHospital
from aggregator import WorkerResearchCenter
from inference import predict_and_generate_report

# Execution Code: Include the main workflow logic here

if __name__ == "__main__":
    survival_info_df = pd.read_csv('/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv')
    name_mapping_df = pd.read_csv('/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv')

    name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True)
    df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")

    paths = []
    for _, row in df.iterrows():
        id_ = row['Brats20ID']
        phase = id_.split("_")[-2]
        if phase == 'Training':
            path = os.path.join(config.train_root_dir, id_)
        else:
            path = os.path.join(config.test_root_dir, id_)
        paths.append(path)

    df['path'] = paths

    train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
    train_data["Age_rank"] = train_data["Age"] // 10 * 10
    train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=7, random_state=config.seed, shuffle=True)
    for i, (train_index, val_index) in enumerate(skf.split(train_data, train_data["Age_rank"])):
        train_data.loc[val_index, "fold"] = i

    train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
    val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)

    train_data.to_csv("train_data.csv", index=False)

    # Simulate local training and parameter sharing
    clients = [
        ClientHospital("Hospital A", DataLoader([]), SimpleModel()),
        ClientHospital("Hospital B", DataLoader([]), SimpleModel())
    ]

    workers = [
        WorkerResearchCenter("Research Center A")
    ]

    client_hashes = []
    for client in clients:
        hash_link = client.share_parameters()
        client_hashes.append(hash_link)

    # Aggregate parameters at a research center
    for worker in workers:
        worker.aggregate_parameters(client_hashes)

    # Fetch and verify the global model
    global_model_hash = globalSC.get_global_model_hash()
    aggregated_model_data = get_from_ipfs(global_model_hash)
    print("Aggregated Model Data:", aggregated_model_data)

    # Predict and generate a report using the aggregated global model
    image_path = 'path_to_brain_tumor_image.jpg'  # Replace with your image file path
    model = SimpleModel()  # Use the SimpleModel or your actual trained model
    predict_and_generate_report(model, image_path, global_model_hash)
