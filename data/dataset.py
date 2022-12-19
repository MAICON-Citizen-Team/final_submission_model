import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset
from module.transform import test_transform, train_transform


class IRDataset(Dataset):
    def __init__(self, path, transform=train_transform):
        self.path = path
        self.transform = transform

    def __getitem__(self, idx):
        noise_image = cv2.imread(self.path[idx][0])
        gt_image = cv2.imread(self.path[idx][1])

        noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        gt_image = torch.tensor(gt_image.transpose(2, 0, 1))

        data_sample = { "image" : noise_image }
        if self.transform:
            data_sample = self.transform(**data_sample)
        data_sample["gt"] = gt_image
            
        return data_sample

    def __len__(self):
        return len(self.path)

    
class IRTestDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.transform = test_transform

    def __getitem__(self, idx):
        noise_image = cv2.imread(self.path[idx])
        noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)
        image_id = int(self.path[idx].split(os.path.sep)[-1].split("_")[-1].split(".")[0])
        
        if self.transform:
            noise_image = self.transform(image=noise_image)["image"]
        data_sample = {"image" : noise_image, "id" : image_id}

        return data_sample

    def __len__(self):
        return len(self.path)
