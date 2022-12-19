from datetime import datetime
import gc
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import PeakSignalNoiseRatio

from config import get_config
from module.utils import fix_seed, count_parameters
from module.loss import PSNRLoss
from model import NAFNet
from data import create_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = (torch.tensor((0.359, 0.370, 0.361), dtype=float).view(3, 1, 1) * 255).to(device)
STD = (torch.tensor((0.087, 0.086, 0.088), dtype=float).view(3, 1, 1) * 255).to(device)

    
def train(model, config):
    # Prepare for training
    fix_seed(config["seed"])
    run_id = datetime.now().strftime("%m%d_%H%M")


    # Data Preparation
    train_dataset, val_dataset = create_dataset(config)

    train_dataloader = DataLoader(train_dataset, **config["data"]["dataloader"]["train"])     
    val_dataloader = DataLoader(val_dataset, **config["data"]["dataloader"]["val"])

    # Training component
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), **config["optimizer"]["Adam"])
    scheduler = LinearLR(optimizer, **config["scheduler"]["LinearLR"])
    criterion = PSNRLoss().to(device)
    metric = PeakSignalNoiseRatio().to(device)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    print("START TRAINING\n")
    progress = tqdm(total=config["eval_step"], leave=False)
    train_loss = torch.zeros(1, dtype=torch.float).to(device)
    train_iter = 0
    best_score = 0
    
    while True:
        # Train
        for data in train_dataloader:
            if train_iter >= config["iteration"]:
                del model
                progress.close()
                print("END TRAINING")
                return 
            progress.update()

            model.train()
            with torch.cuda.amp.autocast():
                noise_image = data["image"].to(device, non_blocking=True)
                gt_image = data["gt"].to(device, non_blocking=True).float()

                pred = model(noise_image)
                pred = pred * STD + MEAN
    
                loss = criterion(pred, gt_image) 
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss
            train_iter += 1

            if train_iter % config["eval_step"] == 0 or train_iter >= config["iteration"]:
                del data, noise_image, #gt_image, pred
                train_loss = (train_loss / config["eval_step"]).item()
                val_score = validate(model, val_dataloader, metric=metric)
                
                print(f"Train Iteration {train_iter} | Train Loss: {train_loss :.2f}dB   Validation Score {val_score :.2f}dB")

                if val_score > best_score:
                    best_score = val_score
                    torch.save(model.state_dict(), f"save/{run_id}_{train_iter}.pth")
                    print("Model with best score saved\n")

                train_loss *= 0
                progress.reset()

            # Optimize memory
            gc.collect()

        scheduler.step()   

def validate(model, dataloader, metric):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    val_score = []
    
    for data in dataloader:
        with torch.no_grad():
            noise_image = data["image"].to(device, non_blocking=True)
            target = data["gt"].to(device, non_blocking=True).float()
            
            pred = model(noise_image)
            pred = pred * STD + MEAN
            val_score.append(metric(pred, target).item())

        # Optimize memory 
        gc.collect()

    del data, noise_image #, gt_image, pred
    return np.mean(val_score)

if __name__ == "__main__":
    config = get_config()
    model = NAFNet(**config["model"])

    train(model, config=config)