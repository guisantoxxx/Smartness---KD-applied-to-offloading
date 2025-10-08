# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import (InstanceList, OptInstanceList, OptConfigType, reduce_mean)
from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector

# Guilherme: Este código define o comportamento do forward e da loss para modelos CrossKD, conforme descrito no paper.
# Guilherme: Basicamente, ele implementa o funcionamento do modelo estudante, incluindo toda a lógica do CrossKD.
# O professor (teacher) fica congelado e é usado apenas para operações relacionadas à distilação.

# Guilherme: Para cada tipo de detector diferente, ele cria um arquivo de CrossKD diferente, para se adaptar para os diferentes detecctors

@MODELS.register_module()
class CrossKDATSS(CrossKDSingleStageDetector):

    def __init__(self, 
                 kd_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(kd_cfg=kd_cfg,**kwargs)
        # Initialize centerness KD loss if specified in kd_cfg
        # (Paper: centerness distillation)
        self.loss_center_kd = None
        if kd_cfg.get('loss_center_kd', None):
            self.loss_center_kd = MODELS.build(kd_cfg['loss_center_kd'])
                
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.
        
        1. Extract teacher features
        2. Extract student features
        3. Forward both up to distillation points
        4. Reuse teacher head to get student predictions aligned
        5. Compute original and KD losses
        """
        # 1. Teacher feature extraction (Paper: teacher forward)
        tea_x = self.teacher.extract_feat(batch_inputs)
        # Forward through head for teacher, capture features at reuse index
        tea_cls_scores, tea_bbox_preds, tea_centernesses, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_hkd_single, 
                        tea_x,
                        self.teacher.bbox_head.scales, 
                        module=self.teacher)
            
        # 2. Student feature extraction (Paper: student forward)
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds, stu_centernesses, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_hkd_single, 
                        stu_x,
                        self.bbox_head.scales, 
                        module=self)
            
        # 3. Reuse teacher head parts for distillation (Paper: head reuse)
        reused_cls_scores, reused_bbox_preds, reused_centernesses = multi_apply(
            self.reuse_teacher_head, 
            tea_cls_hold,  # teacher features at reuse index
            tea_reg_hold, 
            stu_cls_hold,  # student features at reuse index
            stu_reg_hold, 
            self.teacher.bbox_head.scales)

        # Unpack ground truth for loss computation
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        # 4. Compute all losses including KD (Paper: loss functions)
        losses = self.loss_by_feat(tea_cls_scores, # Teacher results are used in the loss func against reused results (teacher pred with intermediate features from student)
                                   tea_bbox_preds,
                                   tea_centernesses,
                                   tea_x,
                                   stu_cls_scores, # Student results are compared against gt
                                   stu_bbox_preds,
                                   stu_centernesses,
                                   stu_x,
                                   reused_cls_scores,
                                   reused_bbox_preds,
                                   reused_centernesses,
                                   batch_gt_instances,
                                   batch_img_metas, 
                                   batch_gt_instances_ignore)
        return losses
    
    # This function defines a single forward for either the student or the teacher
    def forward_hkd_single(self, x, scale, module):
        """
        Forward pass up to distillation layer:
        - Pass through classification and regression convs
        - Capture features at reuse index
        """
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        # Iterate classification convs
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            #  Capture feature for reuse at paper-specified layer
            # Guilherme: In the paper, they state that they use the student's intermediate features from layer i to give it to
            # teacher head layer i + 1 to continue the forward. This if checks the index of the layer and saves the intermediate features if its the 
            # last student layer used for distillation
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat  # (Paper: capture feature)
            # Classification intermediate features
            cls_feat = cls_conv.activate(cls_feat)
        # Iterate regression convs similarly
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            # Bb intermediate features
            reg_feat = reg_conv.activate(reg_feat)
        # Original student/teacher predictions
        cls_score = module.bbox_head.atss_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.atss_reg(reg_feat)).float()
        centerness = module.bbox_head.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness, cls_feat_hold, reg_feat_hold
    
    # Guilherme: This is the function responsible for using the student intermediate features to generate a prediction via teacher head
    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        """
        Reuse teacher head layers on student features:
        - Align student→teacher stats
        - Continue convolutions from reuse index
        """
        #  Align statistics (Paper: feature alignment)
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        # Optional ReLU as in teacher head
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        module = self.teacher.bbox_head
        # Continue remaining conv layers from teacher head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        # Get predictions for KD
        reused_cls_score = module.atss_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.atss_reg(reused_reg_feat)).float()
        reused_centerness = module.atss_centerness(reused_reg_feat)
        return reused_cls_score, reused_bbox_pred, reused_centerness
    
    def align_scale(self, stu_feat, tea_feat):
        """
        Align student feature statistics to teacher:
        - Normalize student
        - Apply teacher mean/std
        (Paper: mean-std feature transformation)
        """
        N, C, H, W = stu_feat.size()
        # normalize student feature across spatial dims
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        # get teacher stats
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        # apply teacher stats to student
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
    
    # Guilherme: Se fossemos alterar alguma parte para adaptar a nosso problema, provavelmente seria aqui
    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_centernesses: List[Tensor],
            tea_feats: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            centernesses: List[Tensor],
            feats: List[Tensor],
            reused_cls_scores: List[Tensor],
            reused_bbox_preds: List[Tensor],
            reused_centernesses: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """
        Compute loss components:
        - Original detection losses
        - KD losses on predictions
        - (Optional) feature distillation
        """
        # Standard ATSS losses (Paper: detection baseline losses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=cls_scores[0].device)
        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets
        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=cls_scores[0].device)).item()

        # Traditional losses, calculated using students prediction and gt
        losses_cls, losses_bbox, loss_centerness, \
            bbox_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores, # Student classification score
                bbox_preds, # Student Bounding Box prediction
                centernesses, # Student centernesses
                labels_list, # Gt classes
                label_weights_list, 
                bbox_targets_list, # Gt bounding boxes
                avg_factor=avg_factor)

        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_centerness=loss_centerness)

        # KD losses on predictions (Paper: pred imitation losses)
        # Guilherme: Calculate the KD loss, between teachers prediction and prediction using intermediate features
        losses_cls_kd, losses_reg_kd, losses_center_kd = multi_apply(
            self.pred_imitation_loss_single,
            labels_list,
            anchor_list,
            tea_cls_scores, # Teacher classification scores
            tea_bbox_preds, # Teacher BB
            tea_centernesses, # Teacher certerness
            reused_cls_scores, # Intermediate features based
            reused_bbox_preds,# Intermediate features based
            reused_centernesses,# Intermediate features based
            label_weights_list,
            avg_factor=avg_factor)
        losses.update(dict(loss_cls_kd=losses_cls_kd, loss_reg_kd=losses_reg_kd, losses_center_kd=losses_center_kd))
        
        # Optional feature distillation (Paper: feature-level KD)
        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(feats, tea_feats)
            ]
            losses.update(loss_feat_kd=losses_feat_kd)
        return losses
    
    # Guilherme: This function is the one responsible for calculating the loss between teacher and intermediate features based prediction
    def pred_imitation_loss_single(self, 
                                   labels,
                                   anchors,
                                   tea_cls_score, 
                                   tea_bbox_pred,
                                   tea_centernesses,
                                   reused_cls_score, 
                                   reused_bbox_pred,
                                   reused_centernesses,
                                   label_weights, 
                                   avg_factor):
        """
        Compute per-anchor prediction imitation losses:
        - Classification KD
        - Regression KD
        - Centerness KD
        """
        # ---- Classification branch KD ---- (Paper eq. for cls KD)
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
        label_weights = label_weights.reshape(-1)
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # ---- Regression branch KD ---- (Paper eq. for box KD)
        bbox_coder = self.bbox_head.bbox_coder
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3, 1).reshape(-1, bbox_coder.encode_size)
        reused_bbox_pred = reused_bbox_pred.permute(0, 2, 3, 1).reshape(-1, bbox_coder.encode_size)
        anchors = anchors.reshape(-1, anchors.size(-1))
        tea_bbox_pred = bbox_coder.decode(anchors, tea_bbox_pred)
        reused_bbox_pred = bbox_coder.decode(anchors, reused_bbox_pred)
        # weight by teacher confidence
        reg_weights = tea_cls_score.max(dim=1)[0].sigmoid()
        reg_weights[label_weights == 0] = 0
        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_pred,
            tea_bbox_pred,
            weight=reg_weights,
            avg_factor=avg_factor)
        
        # ---- Centerness branch KD ---- (Paper eq. for centerness KD)
        labels = labels.reshape(-1)
        bg_class_ind = self.bbox_head.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        tea_centernesses = tea_centernesses.permute(0, 2, 3, 1).reshape(-1)
        reused_centernesses = reused_centernesses.permute(0, 2, 3, 1).reshape(-1)

        if len(pos_inds) > 0:
            loss_center_kd = self.loss_center_kd(
                reused_centernesses[pos_inds],
                tea_centernesses[pos_inds].sigmoid(),
                avg_factor=avg_factor)
        else:
            loss_center_kd = reused_centernesses.new_tensor(0.)
        return loss_cls_kd, loss_reg_kd, loss_center_kd
