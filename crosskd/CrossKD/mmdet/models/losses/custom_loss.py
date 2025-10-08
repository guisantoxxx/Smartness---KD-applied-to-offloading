import torch
import torch.nn as nn
from mmdet.registry import MODELS

@MODELS.register_module()
class CustomLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        loss = torch.abs(pred - target)
        loss = loss.mean()
        return loss * self.loss_weight
