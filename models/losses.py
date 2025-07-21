

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math
import numpy as np

import torch
from torch import nn
# from mmdet.models.builder import LOSSES
# from mmdet.models.losses.utils import weighted_loss
from copy import deepcopy
# from .gmm import GaussianMixture

def kld_single2single(g1, g2):
    """Compute Kullback-Leibler Divergence.

    Args:
        g1 (dict[str, torch.Tensor]): Gaussian distribution 1.
        g2 (torch.Tensor): Gaussian distribution 2.

    Returns:
        torch.Tensor: Kullback-Leibler Divergence.
    """
    p_mu = g1.mu
    p_var = g1.var
    assert p_mu.dim() == 3 and p_mu.size()[1] == 1
    assert p_var.dim() == 4 and p_var.size()[1] == 1
    p_mu = p_mu.squeeze(1)
    p_var = p_var.squeeze(1)
    t_mu, t_var = g2
    delta = (p_mu - t_mu).unsqueeze(-1)
    t_inv = torch.inverse(t_var)
    term1 = delta.transpose(-1, -2).matmul(t_inv).matmul(delta).squeeze(-1)
    term2 = torch.diagonal(
        t_inv.matmul(p_var),
        dim1=-2,
        dim2=-1).sum(dim=-1, keepdim=True) + \
        torch.log(torch.det(t_var) / torch.det(p_var)).reshape(-1, 1)

    return 0.5 * (term1 + term2) - 1


#@weighted_loss
def kld_loss1(pred, target, eps=1e-6):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Convexes with shape (N, 9, 2).
        target (torch.Tensor): Polygons with shape (N, 4, 2).
        eps (float): Defaults to 1e-6.

    Returns:
        torch.Tensor: Kullback-Leibler Divergence loss.
    """
    pred = pred.reshape(-1, 9, 2)
    target = target.reshape(-1, 4, 2)

    assert pred.size()[0] == target.size()[0] and target.numel() > 0
    gmm = GaussianMixture(n_components=1, requires_grad=True)
    gmm.fit(pred)
    kld = kld_single2single(gmm, gt2gaussian(target))
    kl_agg = kld.clamp(min=eps)
    loss = 1 - 1 / (2 + torch.sqrt(kl_agg))

    return loss

class KLDRepPointsLoss(nn.Module):
    """Kullback-Leibler Divergence loss for RepPoints.

    Args:
        eps (float): Defaults to 1e-6.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    """

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(KLDRepPointsLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * kld_loss1(
            pred,
            target,
            # weight,
            eps=self.eps
            # reduction=reduction,
            # avg_factor=avg_factor,
            # **kwargs
        )
        return loss_bbox

def gt2gaussian(target):
    """Convert polygons to Gaussian distributions.

    Args:
        target (torch.Tensor): Polygons with shape (N, 8).

    Returns:
        dict[str, torch.Tensor]: Gaussian distributions.
    """
    L = 3
    center = torch.mean(target, dim=1)
    edge_1 = target[:, 1, :] - target[:, 0, :]
    edge_2 = target[:, 2, :] - target[:, 1, :]
    w = (edge_1 * edge_1).sum(dim=-1, keepdim=True)
    w_ = w.sqrt()
    h = (edge_2 * edge_2).sum(dim=-1, keepdim=True)
    diag = torch.cat([w, h], dim=-1).diag_embed() / (4 * L * L)
    cos_sin = edge_1 / w_
    neg = torch.tensor([[1, -1]], dtype=torch.float32).to(cos_sin.device)
    R = torch.stack([cos_sin * neg, cos_sin[..., [1, 0]]], dim=-2)

    return (center, R.matmul(diag).matmul(R.transpose(-1, -2)))





def xy_wh_r_2_xy_sigma(xywhr):
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-6).reshape(-1, 2)  #改--lm  min=1e-7, max=1e7
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(
        _shape[:-1] + (2, 2))

    return xy, sigma


def xy_stddev_pearson_2_xy_sigma(xy_stddev_pearson):
    _shape = xy_stddev_pearson.shape
    assert _shape[-1] == 5
    xy = xy_stddev_pearson[..., :2]
    stddev = xy_stddev_pearson[..., 2:4]
    pearson = xy_stddev_pearson[..., 4].clamp(min=1e-7 - 1, max=1 - 1e-7)
    covar = pearson * stddev.prod(dim=-1)
    var = stddev.square()
    sigma = torch.stack((var[..., 0],
                         covar,
                         covar,
                         var[..., 1]), dim=-1).reshape(_shape[:-1] + (2, 2))
    return xy, sigma


def postprocess(distance, fun='log1p', tau=1.0):
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


# @weighted_loss
def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """
    given any positive-definite symmetrical 2*2 matrix Z:
    Tr(Z^(1/2)) = sqrt(λ_1) + sqrt(λ_2)
    where λ_1 and λ_2 are the eigen values of Z

    meanwhile we have:
    Tr(Z) = λ_1 + λ_2
    det(Z) = λ_1 * λ_2

    combination with following formula:
    (sqrt(λ_1) + sqrt(λ_2))^2 = λ_1 + λ_2 + 2 * sqrt(λ_1 * λ_2)

    yield:
    Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z)))

    for gwd loss the frustrating coupling part is:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))

    assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then:
    Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2))
    = Tr(Σp^(1/2) * Σp^(1/2) * Σt)
    = Tr(Σp * Σt)
    det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2))
    = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2))
    = det(Σp * Σt)

    and thus we can rewrite the coupling part as:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z)))
    = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt)))
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(dim1=-2, dim2=-1).sum(
        dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(0).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

    if normalize:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


# @weighted_loss
def kld_loss(xy_p, Sigma_p,xy_t,Sigma_t,fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
# def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):   #lmgai  22-8-21
    # todo
    xy_p, Sigma_p = xy_p, Sigma_p
    xy_t, Sigma_t = xy_t, Sigma_t

    _shape = xy_p.shape

    # xy_p = xy_p.reshape(-1, 2)
    # xy_t = xy_t.reshape(-1, 2)
    # Sigma_p = Sigma_p.reshape(-1, 2, 2)
    # Sigma_t = Sigma_t.reshape(-1, 2, 2)

    xy_p = xy_p.reshape(-1, 2).float()
    xy_t = xy_t.reshape(-1, 2).float()
    Sigma_p = Sigma_p.reshape(-1, 2, 2).float()
    Sigma_t = Sigma_t.reshape(-1, 2, 2).float()

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)

    # from apex import amp
    # amp.register_float_function(torch, 'det')

    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)   #det(x)  返回X的行列式

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(
        dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(
        Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    Sigma_t_det_log = torch.where(torch.isnan(Sigma_t_det_log), torch.full_like(Sigma_t_det_log, 0), Sigma_t_det_log)   #如果为nan，则为0
    Sigma_p_det_log = torch.where(torch.isnan(Sigma_p_det_log), torch.full_like(Sigma_p_det_log, 0),Sigma_p_det_log)  # 如果为nan，则为0
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    distance = distance.reshape(_shape[:-1])

    return postprocess(distance, fun=fun, tau=tau)


# @weighted_loss
def jd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    jd = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=False,
                  reduction='none')
    jd = jd + kld_loss(target, pred, fun='none', tau=0, alpha=alpha,
                       sqrt=False,
                       reduction='none')
    jd = jd * 0.5
    if sqrt:
        jd = jd.clamp(0).sqrt()
    return postprocess(jd, fun=fun, tau=tau)


# @weighted_loss
def kld_symmax_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    kld_pt = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_tp = kld_loss(target, pred, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_symmax = torch.max(kld_pt, kld_tp)
    return postprocess(kld_symmax, fun=fun, tau=tau)


# @weighted_loss
def kld_symmin_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    kld_pt = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_tp = kld_loss(target, pred, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_symmin = torch.min(kld_pt, kld_tp)
    return postprocess(kld_symmin, fun=fun, tau=tau)


# @LOSSES.register_module()
class GDLoss(nn.Module):
    BAG_GD_LOSS = {'gwd': gwd_loss,
                   'kld': kld_loss,
                   'jd': jd_loss,
                   'kld_symmax': kld_symmax_loss,
                   'kld_symmin': kld_symmin_loss}
    BAG_PREP = {'xy_stddev_pearson': xy_stddev_pearson_2_xy_sigma,
                'xy_wh_r': xy_wh_r_2_xy_sigma}

    def __init__(self, loss_type, representation='xy_stddev_pearson',
                 fun='log1p', tau=0.0, alpha=1.0, reduction='mean',
                 loss_weight=1.0, **kwargs):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'none']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = self.BAG_PREP[representation]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        pred = self.preprocess(pred)
        target = self.preprocess(target)

        return self.loss(
            pred,
            target,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            **_kwargs) * self.loss_weight




# class IOUloss(nn.Module):
#     def __init__(self, reduction="none", loss_type="iou"):
#         super(IOUloss, self).__init__()
#         self.reduction = reduction
#         self.loss_type = loss_type
#
#     def forward(self, pred, target):
#         assert pred.shape[0] == target.shape[0]
#
#         pred = pred.view(-1, 4)
#         target = target.view(-1, 4)
#         tl = torch.max(
#             (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
#         )
#         br = torch.min(
#             (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
#         )
#
#         area_p = torch.prod(pred[:, 2:], 1)
#         area_g = torch.prod(target[:, 2:], 1)
#
#         en = (tl < br).type(tl.type()).prod(dim=1)
#         area_i = torch.prod(br - tl, 1) * en
#         iou = (area_i) / (area_p + area_g - area_i + 1e-16)
#
#         if self.loss_type == "iou":
#             loss = 1 - iou ** 2
#         elif self.loss_type == "giou":
#             c_tl = torch.min(
#                 (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
#             )
#             c_br = torch.max(
#                 (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
#             )
#             area_c = torch.prod(c_br - c_tl, 1)
#             giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
#             loss = 1 - giou.clamp(min=-1.0, max=1.0)
#
#         if self.reduction == "mean":
#             loss = loss.mean()
#         elif self.reduction == "sum":
#             loss = loss.sum()
#
#         return loss




# # 计算旋转矩形iou   #未使用  ---lm
# def rotate_box_iou(box1, box2, GIoU=False):
#     """
#     计算box1中的所有box 与 box2中的所有box的旋转矩形iou （1对1）
#     :param box1: GT tensor  size=(n, [xywhθ])
#     :param box2: anchor  size= (n, [xywhθ])
#     :param GIoU: 是否使用GIoU的标志位
#     :return:
#              box2所有box与box1的IoU  size= (n)
#     """
#     ft = torch.cuda.FloatTensor
#     if isinstance(box1, list):  box1 = ft(box1)
#     if isinstance(box2, list):  box2 = ft(box2)
#
#     if len(box1.shape) < len(box2.shape):  # 输入的单box维度不匹配时，unsqueeze一下 确保两个维度对应两个维度
#         box1 = box1.unsqueeze(0)
#     if len(box2.shape) < len(box1.shape):  # 输入的单box维度不匹配时，unsqueeze一下 确保两个维度对应两个维度
#         box2 = box2.unsqueeze(0)
#     if not box1.shape == box2.shape:  # 若两者num数量不等则报错
#         print('计算旋转矩形iou时有误，输入shape不相等')
#         print('----------------box1:--------------------')
#         print(box1.shape)
#         print(box1)
#         print('----------------box2:--------------------')
#         print(box2.shape)
#         print(box2)
#     # print(box1)
#     # box(n, [xywhθ])
#     box1 = box1[:, :5]
#     box2 = box2[:, :5]
#
#     if GIoU:
#         mode = 'giou'
#     else:
#         mode = 'iou'
#
#     ious = []
#     for i in range(len(box2)):
#         # print(i)
#         r_b1 = get_rotated_coors(box1[i])
#         r_b2 = get_rotated_coors(box2[i])
#
#         ious.append(skewiou(r_b1, r_b2, mode=mode))
#
#     # if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
#     #     c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
#     #     c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
#     #     c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
#     #     return iou - (c_area - union_area) / c_area  # GIoU
#     # print(ious)
#     return ft(ious)