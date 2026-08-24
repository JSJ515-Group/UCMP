# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy

import numpy as np
import torch
from timm.models.layers import DropPath
from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd

from mmseg.models import build_segmentor
from mmseg.models.uda.uda_decorator import get_module
import math


class EMATeacher(Module):

    def __init__(self, use_mask_params, cfg):
        super(EMATeacher, self).__init__()
        prefix = 'mask_' if use_mask_params else ''
        self.alpha = cfg[f'{prefix}alpha']
        if self.alpha == 'same':
            self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg[f'{prefix}pseudo_threshold']
        if self.pseudo_threshold == 'same':
            self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']

        self.legacy_scalar_pseudo_weight = cfg.get(
            'legacy_scalar_pseudo_weight', True)

        self.unc_cfg = cfg.get('uncertainty_weight', {}) or {}
        self.unc_enable = bool(self.unc_cfg.get('enable', False))
        self.unc_type = self.unc_cfg.get('type', 'entropy')
        assert self.unc_type in ['entropy', 'conf']
        self.unc_alpha = float(self.unc_cfg.get('alpha', 5.0))
        self.unc_gamma = float(self.unc_cfg.get('gamma', 2.0))
        self.unc_w_min = float(self.unc_cfg.get('w_min', 0.05))
        self.unc_use_exp = bool(self.unc_cfg.get('use_exp', True))
        self.mean_calib_cfg = cfg.get('mean_calibration', {}) or {}
        self.mean_calib_enable = bool(self.mean_calib_cfg.get('enable', False))
        self.calib_gain = float(cfg.get('calib_gain', 1.0))

        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        self.debug = False
        self.debug_output = {}

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self, model):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    model.parameters()):
            if not param.data.shape:
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_debug_state(self):
        self.get_ema_model().automatic_debug = False
        self.get_ema_model().debug = self.debug

    @torch.no_grad()
    def _compute_uncertainty_weight(self, prob, conf):
        if not self.unc_enable:
            return torch.ones_like(conf)

        if self.unc_type == 'entropy':
            eps = 1e-8
            ent = -(prob * (prob.clamp_min(eps).log())).sum(dim=1)
            ent_norm = ent / math.log(prob.size(1))
            if self.unc_use_exp:
                w = torch.exp(-self.unc_alpha * ent_norm)
            else:
                w = (1.0 - ent_norm).clamp_min(0.0) ** self.unc_gamma
        elif self.unc_type == 'conf':
            w = conf.clamp(0.0, 1.0) ** self.unc_gamma
        else:
            w = torch.ones_like(conf)

        w = w.clamp_min(self.unc_w_min).clamp_max(1.0)
        return w

    def get_pseudo_label_and_weight(self, logits):
        """
          pseudo_label:  [B,H,W]
          pseudo_weight: [B,H,W]
        """
        with torch.no_grad():
            prob = torch.softmax(logits.detach(), dim=1)
            conf, pseudo_label = torch.max(prob, dim=1)

            if self.legacy_scalar_pseudo_weight:
                if self.pseudo_threshold is not None:
                    r = (conf >= float(self.pseudo_threshold)).float().mean()
                    pseudo_weight = r * torch.ones_like(conf)
                else:
                    pseudo_weight = torch.ones_like(conf)
                return pseudo_label, pseudo_weight

            if self.pseudo_threshold is not None:
                r = (conf >= float(self.pseudo_threshold)).float().mean()
            else:
                r = torch.tensor(1.0, device=conf.device)

            if not self.unc_enable:
                pseudo_weight = r * torch.ones_like(conf)
                return pseudo_label, pseudo_weight

            w_pixel = self._compute_uncertainty_weight(prob, conf)
            B = conf.shape[0]
            if self.pseudo_threshold is not None:
                r_img = (conf >= float(self.pseudo_threshold)).float().view(B, -1).mean(dim=1)
            else:
                r_img = torch.ones(B, device=conf.device)

            r_img_map = r_img.view(B, 1, 1)

            if not self.mean_calib_enable:

                pseudo_weight = r_img_map * w_pixel
            else:
                mean_w_img = w_pixel.view(B, -1).mean(dim=1).clamp_min(1e-6)
                scale = (self.calib_gain * r_img / mean_w_img).view(B, 1, 1)
                pseudo_weight = w_pixel * scale
                pseudo_weight = pseudo_weight.clamp_(0.0, 2.0)

            return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def update_weights(self, model, iter):
        if iter == 0:
            self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, iter)

    def __call__(self, target_img, target_img_metas, valid_pseudo_mask):
        self.update_debug_state()

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().generate_pseudo_label(
            target_img, target_img_metas)

        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
            ema_logits)
        del ema_logits

        pseudo_weight = self.filter_valid_pseudo_region(
            pseudo_weight, valid_pseudo_mask)

        self.debug_output = self.ema_model.debug_output
        return pseudo_label, pseudo_weight
