import copy
import os
import time
import gc
import warnings
import numpy as np
import pandas as pd
from random import randint
from tqdm import tqdm
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skimage.transform import resize
from skimage.util import montage
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from IPython.display import Image as show_gif
import imageio
import nibabel as nib
import requests  # For IPFS and Blockchain interactions
import json  # For serializing and deserializing model parameters
from PIL import Image
import openai

warnings.simplefilter("ignore")

# Global IPFS and Blockchain settings
IPFS_API_URL = "http://localhost:5001/api/v0"

class GlobalConfig:
    root_dir = '/home/zengsn/BRATS/Data/Brats2020/'
    train_root_dir = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    path_to_csv = 'train_data.csv'
    pretrained_model_path = None  # path to pre-trained UNet model
    ae_pretrained_model_path = None  # path to pre-trained Autoencoder model
    train_logs_path = 'train_log.csv'
    seed = 55

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = GlobalConfig()
seed_everything(config.seed)

# IPFS functions
def add_to_ipfs(data):
    res = requests.post(f"{IPFS_API_URL}/add", files={"file": data.encode('utf-8')})
    return res.json()["Hash"]

def get_from_ipfs(hash_link):
    res = requests.post(f"{IPFS_API_URL}/cat?arg={hash_link}")
    return res.text

# Blockchain Smart Contract (Placeholder for actual implementation)
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

globalSC = SmartContract()  # Global Smart Contract for model management
rewardSC = SmartContract()  # Reward Smart Contract

# Utility Classes and Functions
class Image3dToGIF3d:
    def __init__(self, img_dim=(55, 55, 55), figsize=(15, 10), binary=False, normalizing=True):
        self.img_dim = img_dim
        self.figsize = figsize
        self.binary = binary
        self.normalizing = normalizing

    def _explode(self, data):
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def _expand_coordinates(self, indices):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    def _normalize(self, arr):
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)

    def _scale_by(self, arr, factor):
        mean = np.mean(arr)
        return (arr - mean) * factor + mean

    def get_transformed_data(self, data):
        if self.binary:
            resized_data = resize(data, self.img_dim, preserve_range=True)
            return np.clip(resized_data.astype(np.uint8), 0, 1).astype(np.float32)

        norm_data = np.clip(self._normalize(data) - 0.1, 0, 1) ** 0.4
        scaled_data = np.clip(self._scale_by(norm_data, 2) - 0.1, 0, 1)
        resized_data = resize(scaled_data, self.img_dim, preserve_range=True)

        return resized_data

    def plot_cube(self, cube, title='', init_angle=0, make_gif=False, path_to_save='filename.gif'):
        if self.binary:
            facecolors = cm.winter(cube)
        else:
            if self.normalizing:
                cube = self._normalize(cube)
            facecolors = cm.gist_stern(cube)

        facecolors[:, :, :, -1] = cube
        facecolors = self._explode(facecolors)
        filled = facecolors[:, :, :, -1] != 0
        x, y, z = self._expand_coordinates(np.indices(np.array(filled.shape) + 1))

        with plt.style.context("dark_background"):
            fig = plt.figure(figsize=self.figsize)
            ax = fig.gca(projection='3d')
            ax.view_init(30, init_angle)
            ax.set_xlim(right=self.img_dim[0] * 2)
            ax.set_ylim(top=self.img_dim[1] * 2)
            ax.set_zlim(top=self.img_dim[2] * 2)
            ax.set_title(title, fontsize=18, y=1.05)
            ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

            if make_gif:
                images = []
                for angle in tqdm(range(0, 360, 5)):
                    ax.view_init(30, angle)
                    fname = str(angle) + '.png'
                    plt.savefig(fname, dpi=120, format='png', bbox_inches='tight')
                    images.append(imageio.imread(fname))
                    os.remove(fname)
                imageio.mimsave(path_to_save, images)
                plt.close()
            else:
                plt.show()

        # Serialize the visual feature and send it to IPFS
        serialized_visual_feature = json.dumps(cube.tolist())
        visual_feature_hash = add_to_ipfs(serialized_visual_feature)
        print(f"Visual feature added to IPFS with hash: {visual_feature_hash}")
        return visual_feature_hash

class ShowResult:
    def mask_preprocessing(self, mask):
        mask = mask.squeeze().cpu().detach().numpy()
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        mask_WT = np.rot90(montage(mask[0]))
        mask_TC = np.rot90(montage(mask[1]))
        mask_ET = np.rot90(montage(mask[2]))

        return mask_WT, mask_TC, mask_ET

    def image_preprocessing(self, image):
        image = image.squeeze().cpu().detach().numpy()
        image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))
        flair_img = np.rot90(montage(image[0]))
        return flair_img

    def plot(self, image, ground_truth, prediction):
        image = self.image_preprocessing(image)
        gt_mask_WT, gt_mask_TC, gt_mask_ET = self.mask_preprocessing(ground_truth)
        pr_mask_WT, pr_mask_TC, pr_mask_ET = self.mask_preprocessing(prediction)

        fig, axes = plt.subplots(1, 2, figsize=(35, 30))

        [ax.axis("off") for ax in axes]
        axes[0].set_title("Ground Truth", fontsize=35, weight='bold')
        axes[0].imshow(image, cmap='bone')
        axes[0].imshow(np.ma.masked_where(gt_mask_WT == False, gt_mask_WT), cmap='cool_r', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_TC == False, gt_mask_TC), cmap='autumn_r', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_ET == False, gt_mask_ET), cmap='autumn', alpha=0.6)

        axes[1].set_title("Prediction", fontsize=35, weight='bold')
        axes[1].imshow(image, cmap='bone')
        axes[1].imshow(np.ma.masked_where(pr_mask_WT == False, pr_mask_WT), cmap='cool_r', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_TC == False, pr_mask_TC), cmap='autumn_r', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_ET == False, pr_mask_ET), cmap='autumn', alpha=0.6)

        plt.tight_layout()
        plt.show()

        # Convert the visual result to a format suitable for LLM (Large Language Model) processing
        serialized_visual_result = json.dumps({
            "image": image.tolist(),
            "ground_truth": {
                "WT": gt_mask_WT.tolist(),
                "TC": gt_mask_TC.tolist(),
                "ET": gt_mask_ET.tolist()
            },
            "prediction": {
                "WT": pr_mask_WT.tolist(),
                "TC": pr_mask_TC.tolist(),
                "ET": pr_mask_ET.tolist()
            }
        })
        visual_result_hash = add_to_ipfs(serialized_visual_result)
        print(f"Visual result added to IPFS with hash: {visual_result_hash}")
        return visual_result_hash

def dice_coef_metric(probabilities: torch.Tensor, truth: torch.Tensor, threshold: float = 0.5, eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= threshold).float()
    assert predictions.shape == truth.shape
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)

def jaccard_coef_metric(probabilities: torch.Tensor, truth: torch.Tensor, threshold: float = 0.5, eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= threshold).float()
    assert predictions.shape == truth.shape

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)

# Dataset and Model Classes
class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "test", is_resize: bool = False):
        self.df = df
        self.phase = phase
        self.augmentations = None  # Placeholder: Define actual augmentations if needed
        self.data_types = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
        self.is_resize = is_resize

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)
            if self.is_resize:
                img = self.resize(img)
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.phase != "test":
            mask_path = os.path.join(root_path, id_ + "_seg.nii.gz")
            mask = self.load_img(mask_path)
            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)
            if self.augmentations:
                augmented = self.augmentations(image=img.astype(np.float32), mask=mask.astype(np.float32))
                img = augmented['image']
                mask = augmented['mask']
            return {"Id": id_, "image": img, "mask": mask}

        return {"Id": id_, "image": img}

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
        return mask

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Training and Evaluation with IPFS and Blockchain
class ClientHospital:
    def __init__(self, name, data, model):
        self.name = name
        self.data = data
        self.model = model
        self.local_params = None

    def local_training(self, epochs=5):
        # Simulate local training
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

class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

def load_model(model_path):
    model = BrainTumorClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_and_generate_report(model, image_path, global_model_hash):
    # Load the global model parameters
    serialized_params = get_from_ipfs(global_model_hash)
    global_model_params = json.loads(serialized_params)
    model.load_state_dict({k: torch.tensor(v) for k, v in global_model_params.items()})

    # Preprocessing the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Model inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        classification_result = 'Tumor' if predicted.item() == 1 else 'No Tumor'

    # OpenAI API call to generate a report
    prompt = f"A brain MRI image has been classified as: {classification_result}. Please provide a detailed report on the implications of this classification."
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=250
    )
    
    report = response.choices[0].text.strip()

    print(f"Classification Result: {classification_result}")
    print(f"Generated Report: {report}")

    # Optionally, save the report to a file
    with open("classification_report.txt", "w") as report_file:
        report_file.write(f"Classification Result: {classification_result}\n\n")
        report_file.write("Generated Report:\n")
        report_file.write(report)

# Execution
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
