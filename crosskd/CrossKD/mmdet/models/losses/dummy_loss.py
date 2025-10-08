from mmdet.registry import MODELS
import torch
import torch.nn as nn

@MODELS.register_module()
class DummyLoss(nn.Module):
    def forward(self, *args, **kwargs):
        device = kwargs.get('device', 'cuda')
        return torch.tensor(0., device=device)
