import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio



class PSNRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.psnr = PeakSignalNoiseRatio()

    def forward(self, pred, target):
        psnr_score = self.psnr(pred, target)
        psnr_loss = -psnr_score

        return psnr_loss
