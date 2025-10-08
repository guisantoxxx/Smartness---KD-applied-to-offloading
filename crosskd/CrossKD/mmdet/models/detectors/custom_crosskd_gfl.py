# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean

from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector


@MODELS.register_module()
class CustomCrossKDGFL(CrossKDSingleStageDetector):

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        tea_x = self.teacher.extract_feat(batch_inputs)
        tea_cls_scores, tea_bbox_preds, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_crosskd_single, tea_x,
                        self.teacher.bbox_head.scales, module=self.teacher)
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_crosskd_single, stu_x,
                        self.bbox_head.scales, module=self)
        reused_cls_scores, reused_bbox_preds = multi_apply(
            self.reuse_teacher_head, tea_cls_hold, tea_reg_hold, stu_cls_hold,
            stu_reg_hold, self.teacher.bbox_head.scales)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        losses = self.loss_by_feat(tea_cls_scores, tea_bbox_preds, tea_x,
                                   stu_cls_scores, stu_bbox_preds, stu_x,
                                   reused_cls_scores, reused_bbox_preds,
                                   batch_gt_instances, batch_img_metas,
                                   batch_gt_instances_ignore)

        device = list(tea_cls_scores)[0].device if len(tea_cls_scores) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        L_ps = torch.tensor(0., device=device)
        for k in ('loss_cls', 'loss_bbox', 'loss_dfl'):
            L_ps = L_ps + self._reduce_item(losses.get(k, None), device)

        '''
        Tentar depois (mais eficiente)
        L_ps = (
            losses['loss_cls'] +
            losses['loss_bbox'] +
            losses['loss_dfl']
        ).sum()
        '''

        # L(p_t, p_gt): soma das componentes do teacher (supondo que já existam no dict) ---
        # Itera sobre as três chaves, somando a média de cada uma delas
        L_pt = torch.tensor(0., device=device)
        for k in ('loss_cls_teacher', 'loss_bbox_teacher', 'loss_dfl_teacher'):
            L_pt = L_pt + self._reduce_item(losses.get(k, None), device)

        # H(p_s): entropia média do student ---
        # concatena logits do student em shape (N_anchors, num_classes)
        student_logits_list = []
        for s in stu_cls_scores:
            n, c, h, w = s.shape
            student_logits_list.append(s.permute(0, 2, 3, 1).reshape(-1, c))
        
        # Entropia esta retornando valores muito pequenos, uma opção é tornar a entropia menor 'dura', ex: Temperature scaling
        if len(student_logits_list) == 0:
            H_ps = torch.tensor(0., device=device)
        else:
            student_logits = torch.cat(student_logits_list, dim=0)  # (num_anchors, C)

            p = torch.sigmoid(student_logits)
            eps = 1e-8
            ent_per_class = -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())  # (num_anchors, C)
            ent_per_anchor = ent_per_class.sum(dim=1) / float(p.shape[1])  # média por classes
            H_ps = ent_per_anchor.mean()  # escalar

        # CQI: pega de kd_cfg se definido, senão escolhe inteiro aleatório [1,15] 
        CQI_val = None
        try:
            CQI_val = float(self.kd_cfg.get('CQI')) if isinstance(self.kd_cfg.get('CQI'), (int, float, str)) else None
        except Exception:
            CQI_val = None

        if CQI_val is None:
            CQI_val = float(torch.randint(1, 16, (1,), device=device).item())
            CQI_val = (16 - CQI_val)  / 15.0  # Inverte o CQI e normaliza

        CQI = torch.tensor(CQI_val, device=device)

        # combina a loss: L_ps + H_ps * CQI * L_pt ---
        final_loss = L_ps + H_ps * CQI * L_pt

        # atualiza o dict para que o runner use essa loss no backward, mantendo infos para log ---
        losses['loss'] = final_loss
        # valores para logging / debugging (detach para não afetar grad)
        losses['student_loss_sum'] = L_ps.detach()
        losses['teacher_loss_sum'] = L_pt.detach()
        losses['student_entropy'] = H_ps.detach()
        losses['CQI'] = CQI.detach()

        return losses
    
    def _reduce_item(self, item, device):
        if item is None:
            return torch.tensor(0., device=device)
        if isinstance(item, (list, tuple)):
            s = torch.tensor(0., device=device)
            for it in item:
                s = s + (it.mean() if isinstance(it, torch.Tensor) else torch.tensor(float(it), device=device))
            return s
        if isinstance(item, torch.Tensor):
            return item.mean()
        # Retorna a média final da loss do batch
        return torch.tensor(float(item), device=device)

    def forward_crosskd_single(self, x, scale, module):
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.gfl_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred, cls_feat_hold, reg_feat_hold

    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        module = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        reused_cls_score = module.gfl_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.gfl_reg(reused_reg_feat)).float()
        return reused_cls_score, reused_bbox_pred

    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_feats: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            feats: List[Tensor],
            reused_cls_scores: List[Tensor],
            reused_bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        # Guilherme: This losses are what we expect, student x gt
        losses_cls, losses_bbox, losses_dfl,\
            new_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.bbox_head.prior_generator.strides,
                avg_factor=avg_factor)

        new_avg_factor = sum(new_avg_factor)
        new_avg_factor = reduce_mean(new_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / new_avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / new_avg_factor, losses_dfl))
        losses = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

        losses_cls_teacher, losses_bbox_teacher, losses_dfl_teacher, kd_avg_factor = multi_apply(
            self.bbox_head.loss_by_feat_single,
            anchor_list,                 # anchors daquele nível
            tea_cls_scores,              # predição de cls do professor
            tea_bbox_preds,              # predição de bbox do professor
            labels_list,                 # labels GT
            label_weights_list,          # pesos dos labels
            bbox_targets_list,           # caixas GT
            self.bbox_head.prior_generator.strides,  # stride de cada nível
            avg_factor=avg_factor
        )

        kd_avg_factor = sum(kd_avg_factor)
        kd_avg_factor = reduce_mean(kd_avg_factor).clamp_(min=1).item()
        losses_bbox_teacher = list(map(lambda x: x / kd_avg_factor, losses_bbox_teacher))
        losses_dfl_teacher = list(map(lambda x: x / kd_avg_factor, losses_dfl_teacher))

        # Atualiza dicionário de losses
        losses.update(
            dict(
                loss_cls_teacher=losses_cls_teacher,
                loss_bbox_teacher=losses_bbox_teacher,
                loss_dfl_teacher=losses_dfl_teacher
            )
        )

        return losses
