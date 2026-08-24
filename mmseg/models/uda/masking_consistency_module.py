# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random

import torch
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd

from mmseg.models.uda.teacher_module import EMATeacher
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform
from mmseg.models.utils.masking_transforms import build_mask_generator


class MaskingConsistencyModule(Module):

    def __init__(self, require_teacher, cfg):
        super(MaskingConsistencyModule, self).__init__()

        self.source_only = cfg.get('source_only', False)
        self.max_iters = cfg['max_iters']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']

        self.mask_mode = cfg['mask_mode']
        self.mask_alpha = cfg['mask_alpha']
        self.mask_pseudo_threshold = cfg['mask_pseudo_threshold']
        self.mask_lambda = cfg['mask_lambda']
        self.mask_gen = build_mask_generator(cfg['mask_generator'])

        assert self.mask_mode in [
            'separate', 'separatesrc', 'separatetrg', 'separateaug',
            'separatesrcaug', 'separatetrgaug'
        ]

        self.proto_cfg = cfg.get('prototype_distill', {}) or {}
        self.proto_enable = bool(self.proto_cfg.get('enable', False))
        self.proto_feat_level = int(self.proto_cfg.get('feat_level', -1))
        self.proto_min_pixels = int(self.proto_cfg.get('min_pixels', 8))
        self.visible_ratio_thr = float(self.proto_cfg.get('visible_ratio_thr', 0.35))
        self.proto_lambda = float(self.proto_cfg.get('proto_lambda', 0.05))
        self.rel_lambda = float(self.proto_cfg.get('rel_lambda', 0.02))
        self.use_cosine = bool(self.proto_cfg.get('use_cosine', True))
        self.ignore_index = 255

        self.teacher = None
        if require_teacher or \
                self.mask_alpha != 'same' or \
                self.mask_pseudo_threshold != 'same':
            self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)

        self.debug = False
        self.debug_output = {}

    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    def update_debug_state(self):
        if self.teacher is not None:
            self.teacher.debug = self.debug

    def _select_feat_tensor(self, feat):
        while isinstance(feat, (list, tuple)):
            feat = feat[self.proto_feat_level]
        return feat

    def _resize_label_weight(self, labels, weights, visible, feat_size):
        # labels: [B,H,W]
        labels_small = F.interpolate(
            labels.unsqueeze(1).float(),
            size=feat_size,
            mode='nearest'
        ).long().squeeze(1)

        if weights is None:
            weights_small = torch.ones_like(labels_small, dtype=torch.float32)
        else:
            if weights.dim() == 4:
                weights = weights.squeeze(1)
            weights_small = F.interpolate(
                weights.unsqueeze(1),
                size=feat_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        if visible is None:
            visible_small = torch.ones_like(labels_small, dtype=torch.float32)
        else:
            if visible.dim() == 4:
                visible = visible.squeeze(1)
            visible_small = F.interpolate(
                visible.unsqueeze(1),
                size=feat_size,
                mode='nearest'
            ).squeeze(1)

        return labels_small, weights_small, visible_small

    def _extract_prototypes(self, feat, labels, weights, visible):
        """
        feat:    [B,C,H,W]
        labels:  [B,H,W]
        weights: [B,H,W]
        visible: [B,H,W]
        """
        B, C, H, W = feat.shape

        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, -1, C)
        label_flat = labels.reshape(B, -1)
        weight_flat = weights.reshape(B, -1)
        visible_flat = visible.reshape(B, -1)

        proto_dicts = []

        for b in range(B):
            proto_dict = {}
            class_ids = torch.unique(label_flat[b])

            for c in class_ids.tolist():
                c = int(c)
                if c == self.ignore_index:
                    continue

                class_mask = (label_flat[b] == c)

                if int(class_mask.sum().item()) < self.proto_min_pixels:
                    continue

                vis_ratio = visible_flat[b][class_mask].float().mean().item()
                if vis_ratio < self.visible_ratio_thr:
                    continue

                eff_weight = weight_flat[b][class_mask] * visible_flat[b][class_mask]
                if float(eff_weight.sum().item()) <= 1e-6:
                    continue

                feat_c = feat_flat[b][class_mask]  # [Nc, C]
                proto_c = (feat_c * eff_weight.unsqueeze(1)).sum(dim=0) / \
                          (eff_weight.sum() + 1e-6)

                proto_dict[c] = proto_c

            proto_dicts.append(proto_dict)

        return proto_dicts

    def _compute_proto_relation_loss(self,
                                     teacher_feat,
                                     student_feat,
                                     pseudo_label,
                                     pseudo_weight,
                                     visible_mask):
        """
        teacher_feat: [B,C,H,W] 
        student_feat: [B,C,H,W] 
        pseudo_label: [B,H,W]
        pseudo_weight:[B,H,W]
        visible_mask: [B,1,H,W] 或 [B,H,W]
        """
        feat_size = student_feat.shape[2:]

        labels_small, weights_small, visible_small = self._resize_label_weight(
            pseudo_label, pseudo_weight, visible_mask, feat_size)

        teacher_visible = torch.ones_like(visible_small)

        teacher_protos = self._extract_prototypes(
            teacher_feat, labels_small, weights_small, teacher_visible)
        student_protos = self._extract_prototypes(
            student_feat, labels_small, weights_small, visible_small)

        proto_losses = []
        rel_losses = []

        for b in range(len(teacher_protos)):
            common_classes = sorted(
                set(teacher_protos[b].keys()) & set(student_protos[b].keys()))

            if len(common_classes) == 0:
                continue

            pt = torch.stack([teacher_protos[b][c] for c in common_classes], dim=0).detach()
            ps = torch.stack([student_protos[b][c] for c in common_classes], dim=0)

            if self.use_cosine:
                proto_loss_b = 1.0 - F.cosine_similarity(ps, pt, dim=1)
                proto_losses.append(proto_loss_b.mean())
            else:
                proto_loss_b = F.mse_loss(ps, pt, reduction='none').mean(dim=1)
                proto_losses.append(proto_loss_b.mean())

            if self.rel_lambda > 0 and len(common_classes) >= 2:
                ps_n = F.normalize(ps, dim=1)
                pt_n = F.normalize(pt, dim=1)

                rs = ps_n @ ps_n.t()
                rt = pt_n @ pt_n.t()

                rel_losses.append(F.l1_loss(rs, rt))

        if len(proto_losses) == 0:
            zero = student_feat.sum() * 0.0
            return zero, zero

        loss_proto = torch.stack(proto_losses).mean() * self.proto_lambda

        if len(rel_losses) == 0:
            loss_rel = student_feat.sum() * 0.0
        else:
            loss_rel = torch.stack(rel_losses).mean() * self.rel_lambda

        return loss_proto, loss_rel

    def __call__(self,
                 model,
                 img,
                 img_metas,
                 gt_semantic_seg,
                 target_img,
                 target_img_metas,
                 valid_pseudo_mask,
                 pseudo_label=None,
                 pseudo_weight=None,
                 ema_model=None):
        self.update_debug_state()
        self.debug_output = {}
        model.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        if not self.source_only:
            if self.teacher is None:
                assert self.mask_alpha == 'same'
                assert self.mask_pseudo_threshold == 'same'
                assert pseudo_label is not None
                assert pseudo_weight is not None
                masked_plabel = pseudo_label
                masked_pweight = pseudo_weight
                teacher_model = ema_model
            else:
                masked_plabel, masked_pweight = \
                    self.teacher(
                        target_img, target_img_metas, valid_pseudo_mask)
                teacher_model = self.teacher.get_ema_model()

                if self.debug:
                    self.debug_output['Mask Teacher'] = {
                        'Img': target_img.detach(),
                        'Pseudo Label': masked_plabel.cpu().numpy(),
                        'Pseudo Weight': masked_pweight.cpu().numpy(),
                    }
        else:
            teacher_model = None

        if self.source_only:
            masked_img = img
            masked_lbl = gt_semantic_seg
            masked_seg_weight = None
            masked_img_metas = img_metas
            visible_mask = None

        elif self.mask_mode in ['separate', 'separateaug']:
            assert img.shape[0] == 2
            masked_img = torch.stack([img[0], target_img[0]])
            masked_lbl = torch.stack(
                [gt_semantic_seg[0], masked_plabel[0].unsqueeze(0)])
            gt_pixel_weight = torch.ones(masked_pweight[0].shape, device=dev)
            masked_seg_weight = torch.stack(
                [gt_pixel_weight, masked_pweight[0]])
            masked_img_metas = [img_metas[0], target_img_metas[0]]
            visible_mask = None

        elif self.mask_mode in ['separatesrc', 'separatesrcaug']:
            masked_img = img
            masked_lbl = gt_semantic_seg
            masked_seg_weight = None
            masked_img_metas = img_metas
            visible_mask = None

        elif self.mask_mode in ['separatetrg', 'separatetrgaug']:
            masked_img = target_img
            masked_lbl = masked_plabel.unsqueeze(1)
            masked_seg_weight = masked_pweight
            masked_img_metas = target_img_metas
            visible_mask = None

        else:
            raise NotImplementedError(self.mask_mode)

        if 'aug' in self.mask_mode:
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            masked_img, _ = strong_transform(
                strong_parameters, data=masked_img.clone())

        if self.mask_gen is not None:
            input_mask = self.mask_gen.generate_mask(masked_img)
            masked_img = masked_img * input_mask
            visible_mask = input_mask
        else:
            visible_mask = torch.ones_like(masked_img[:, :1])

        proto_supported_modes = [
            'separate',
            'separateaug',
            'separatetrg',
            'separatetrgaug',
        ]

        need_proto = (
                self.proto_enable
                and not self.source_only
                and self.mask_mode in proto_supported_modes
                and teacher_model is not None
        )

        if need_proto:
            masked_loss = model.forward_train(
                masked_img,
                masked_img_metas,
                masked_lbl,
                seg_weight=masked_seg_weight,
                return_feat=True,
                return_logits=False,
            )
            student_feat = masked_loss.pop('features')
            student_feat = self._select_feat_tensor(student_feat)
        else:
            masked_loss = model.forward_train(
                masked_img,
                masked_img_metas,
                masked_lbl,
                seg_weight=masked_seg_weight,
            )
            student_feat = None

        if self.mask_lambda != 1:
            masked_loss['decode.loss_seg'] *= self.mask_lambda

        if need_proto:

            if self.mask_mode in ['separate', 'separateaug']:
                proto_student_feat = student_feat[1:2]

                proto_teacher_img = target_img[0:1]

                proto_label = masked_plabel[0:1]
                proto_weight = masked_pweight[0:1]

                proto_visible_mask = visible_mask[1:2]

            elif self.mask_mode in ['separatetrg', 'separatetrgaug']:
                proto_student_feat = student_feat
                proto_teacher_img = target_img
                proto_label = masked_plabel
                proto_weight = masked_pweight
                proto_visible_mask = visible_mask

            else:
                raise RuntimeError(
                    f'Prototype distillation does not support '
                    f'mask_mode={self.mask_mode}'
                )

            for m in teacher_model.modules():
                if isinstance(m, _DropoutNd):
                    m.training = False

                if isinstance(m, DropPath):
                    m.training = False

            with torch.no_grad():
                teacher_feat = teacher_model.extract_feat(
                    proto_teacher_img
                )

                teacher_feat = self._select_feat_tensor(
                    teacher_feat
                )

            assert teacher_feat.shape[0] == proto_student_feat.shape[0], (
                f'Prototype batch mismatch: '
                f'teacher={teacher_feat.shape}, '
                f'student={proto_student_feat.shape}, '
                f'mask_mode={self.mask_mode}'
            )

            assert proto_label.shape[0] == proto_student_feat.shape[0], (
                f'Prototype label batch mismatch: '
                f'label={proto_label.shape}, '
                f'student={proto_student_feat.shape}'
            )

            assert proto_weight.shape[0] == proto_student_feat.shape[0], (
                f'Prototype weight batch mismatch: '
                f'weight={proto_weight.shape}, '
                f'student={proto_student_feat.shape}'
            )

            assert proto_visible_mask.shape[0] == \
                   proto_student_feat.shape[0], (
                f'Prototype visible-mask batch mismatch: '
                f'visible={proto_visible_mask.shape}, '
                f'student={proto_student_feat.shape}'
            )

            loss_proto, loss_rel = \
                self._compute_proto_relation_loss(
                    teacher_feat=teacher_feat,
                    student_feat=proto_student_feat,
                    pseudo_label=proto_label,
                    pseudo_weight=proto_weight,
                    visible_mask=proto_visible_mask,
                )

            masked_loss['loss_proto_distill'] = loss_proto
            masked_loss['loss_relation_distill'] = loss_rel

            if self.debug:
                pass


        if self.debug:
            self.debug_output['Masked'] = model.debug_output
            if masked_seg_weight is not None:
                self.debug_output['Masked']['PL Weight'] = \
                    masked_seg_weight.detach().cpu().numpy()
            if visible_mask is not None:
                self.debug_output['Masked']['Visible Mask'] = \
                    visible_mask.detach().cpu().numpy()

        return masked_loss