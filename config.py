import torch
import numpy as np

class GlobalConfig:
    root_dir = '/home/zengsn/BRATS/Data/Brats2020/'
    train_root_dir = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = '/home/zengsn/BRATS/Data/Brats2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    path_to_csv = 'train_data.csv'
    pretrained_model_path = None
    ae_pretrained_model_path = None
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
