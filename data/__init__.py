import os, random
from glob import glob
import pickle

from .dataset import IRDataset
from module.transform import train_transform


# DATASET_SIZE = 29094
# TRAIN_SIZE = int(DATASET_SIZE * 0.9)


def create_dataset(config, patch=True):
    with open(config["data"]["dataset"]["path"], 'rb') as f:
        train_path, val_path = pickle.load(f)
    
    train_dataset = IRDataset(train_path, transform=train_transform)
    val_dataset = IRDataset(val_path, transform=train_transform)

    return train_dataset, val_dataset


# def filter_path(data_path, data_id):
#     total_path = glob(os.path.join(data_path, "Noised", "*.png"))
#     filtered_path = []

#     for noise_path in total_path:
#         noise_path = os.path.normpath(noise_path).split(os.path.sep)
#         file_id = noise_path.split(os.path.sep)[-1].split("_")[-1].split(".")[0]

#         gt_path = noise_path.split(os.path.sep)
#         gt_path[-2] = "Denoised"
#         gt_path[-1] = f"Denoised_{file_id}.jpg"
#         gt_path = os.path.join(*gt_path)

#         filtered_path.append((noise_path, gt_path))

#     return filtered_path