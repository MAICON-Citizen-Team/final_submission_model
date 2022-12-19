import os, gc, pickle
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio

from config import get_config
from module.utils import fix_seed
from model.naf_net import NAFNet
from data import create_dataset
from data.dataset import IRTestDataset, IRDataset


def test(model, config, result):
    # Prepare for testing
    fix_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MEAN = (torch.tensor((0.359, 0.370, 0.361), dtype=float).view(3, 1, 1) * 255).to(device)
    STD = (torch.tensor((0.087, 0.086, 0.088), dtype=float).view(3, 1, 1) * 255).to(device)
    model = model.to(device)
    metric = PeakSignalNoiseRatio().to(device)

    # Data Preparations
    test_path = sorted(glob(os.path.join("../../data/Test", "*.jpg")))
    test_dataset = IRTestDataset(test_path)
    test_dataloader = DataLoader(test_dataset, **config["data"]["dataloader"]["test"])

    # Testing loop
    print("START TESTING\n")
    model.eval()
    lst = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            noise_image = data["image"].to(device, non_blocking=True)
            image_id = data["id"].to(device, non_blocking=True).item()
            
            pred = model(noise_image)
            if result:
                pred = pred * STD + MEAN
                
                beta = 60 - pred.mean()
                if beta < 0: 
                    beta = 0
                    
                pred = torch.round(torch.clamp(pred + beta, 0, 255))
                pred = pred.cpu().numpy().squeeze(0).transpose(1, 2, 0).astype(np.uint8)
                pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
                
                file_name = os.path.join(config["result_path"], f"Denoised_Inference_{image_id}.jpg")
                cv2.imwrite(file_name, pred)
            
            gc.collect()
            del data, noise_image, pred 
    
    print(np.mean(lst))

    print("END TESTING")


if __name__ == "__main__":
    config = get_config()
    model = NAFNet(**config["model"])
    state_dict = torch.load(f"./save/naf_net.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    test(model, config=config, result=True)