import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from .utils import weighted_loss

@MODELS.register_module()
class BigLoss(nn.Module):
    def __init__(self, cls_weight=1.0, reg_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(self, pred_s, pred_t, gt_labels, gt_bboxes):
        # ------------------------------
        # 1️⃣ Loss do estudante vs GT
        # ------------------------------
        loss_cls = F.cross_entropy(pred_s['cls'], gt_labels)
        loss_reg = F.l1_loss(pred_s['bbox'], gt_bboxes)

        # ------------------------------
        # 2️⃣ Entropia do estudante
        # ------------------------------
        probs_s = F.softmax(pred_s['cls'], dim=1)
        entropy = -torch.sum(probs_s * torch.log(probs_s + 1e-12), dim=1)  # shape: (N,)
        H = entropy.mean()  # escalar médio por batch

        # ------------------------------
        # 3️⃣ CQI aleatório
        # ------------------------------
        CQI = torch.randint(1, 16, (1,), dtype=torch.float, device=pred_s['cls'].device)

        # ------------------------------
        # 4️⃣ Loss do professor vs GT
        # ------------------------------
        loss_teacher_cls = F.cross_entropy(pred_t['cls'], gt_labels)
        loss_teacher_reg = F.l1_loss(pred_t['bbox'], gt_bboxes)

        # ------------------------------
        # 5️⃣ Combina todas
        # ------------------------------
        total_loss = (
            self.cls_weight * loss_cls +
            self.reg_weight * loss_reg +
            H * CQI * (loss_teacher_cls + loss_teacher_reg)
        )

        return total_loss
