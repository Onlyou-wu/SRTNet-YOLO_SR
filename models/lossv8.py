# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics_v8obb import bbox_iou,probiou
from utils.ops import xywh2xyxy, xyxy2xywh
# from ..utils.ops import xywh2xyxy, xyxy2xywh
# from ..utils.metrics import bbox_iou
from .talv8 import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
# from mmcv.ops import box_iou_rotated
from .talv8 import bbox2dist
from .kf_iou_loss import KFLoss
from models.losses import *    #--lmf
# from utils.loss import IOUloss



class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.
        返回左侧和右侧 DFL 损失的总和。

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=True):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)
        self.kfiouLoss= KFLoss(fun='none')
        self.iou_loss = IOUloss(reduction="mean", loss_type="siou")

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """R IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        # loss_iou0 = ((1.0 - iou) * weight).sum() / target_scores_sum
        # print("probiou误差为------:", loss_iou0)

        #siou   #jia --lm
        loss_siou = (self.iou_loss(pred_bboxes[..., :4][fg_mask], target_bboxes[..., :4][fg_mask]))    #Siou use
        # lbox_iou =((1.0 - iou) * weight).sum() / target_scores_sum    #tensor(1.22616, device='cuda:0', grad_fn=<DivBackward0>)
        # print("loss_siou误差为------:", loss_siou)

        # from torch.cuda.amp import autocast
        # with autocast(enabled=False):
        loss_theta01 = self.kfiouLoss(pred_bboxes[fg_mask], target_bboxes[fg_mask], pred_decode=pred_bboxes[fg_mask],
                                 targets_decode=target_bboxes[fg_mask]).reshape(-1, 1) #tensor(0.99924, device='cuda:0')

        loss_theta02 = ((1-loss_theta01) * weight).sum() / target_scores_sum
        # print("kfiouLoss误差为------:", loss_theta02)
        # loss_iou = loss_theta02

        loss_iou = 0.5*loss_siou + 20 * 0.5*loss_theta02
        # loss_iou =  20 * loss_theta02

        #使用probiou和kfiouLoss   24-8-20
        # loss_iou = loss_iou0 + 2*loss_theta02

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        # self.no = 2  # 正确的类别数

        # 调试代码，查看各个张量的尺寸
        print("feats[0].shape[0]:", feats[0].shape[0])
        print("self.no:", self.no)
        for i, xi in enumerate(feats):
            print(f"feats[{i}].numel():", xi.numel())
        print("总元素数量:", sum(xi.numel() for xi in feats))

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = batch   #gai  --lm
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

class v10DetectLoss:
    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], torch.cat((loss_one2many[1], loss_one2one[1]))

class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        # self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=True).to(self.device)


    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
         #预处理目标计数，并与输入的批次大小匹配，输出张量
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    # bboxes[..., :2].mul_(scale_tensor)   # mul_乘  【640，640，640，640】   改lm  xy,w,h,其中宽高不应该乘以640
                    # bboxes[..., :4].mul_(scale_tensor)   # mul_乘  【640，640，640，640】
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]   #得到4参数box和 r
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()   #改变维度
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            # batch_idx = batch["batch_idx"].view(-1, 1)
            # targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            targets = batch

            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()   #*640
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training

            # targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0]])   #gai-lm.4个参数不用乘以640
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        # 只有前四个元素需要缩放
        #--gailm  宽高不乘  X
        # bboxes_for_assigner[..., :2] *= stride_tensor
        #将边界框坐标乘以步幅张量 stride_tensor，恢复到原图尺度。
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss    #fg_mask都是flase的话 就进不去
        if fg_mask.sum():
            # 将目标边界框的坐标除以步幅，恢复到特征图尺度（这里可以看到stride这个参数其实很重要的需要辅助我们真是图和特征图之间相互转化大家需要理解！）
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(   # RotatedBboxLoss
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= 7.5 # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.
        #根据锚点和分布解码预测对象边界框坐标

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


#SIOU
class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="siou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        # pred，target为xywh格式
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        # tl:top_left, br:bottom_right
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )
        # torch.prob为算矩阵乘积，pred[:, 2:]为wh，算出来为面积
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        # en应该是一个比例吧！交集所占两个框所接最小外界矩形面积的比例
        # en = (tl < br).type(tl.type()).prod(dim=1)
        # torch.prod(br - tl, 1)为最小外接矩形的面积，giou需要用到
        hw = (br - tl).clamp(min=0)  # [rows, 2]
        area_i = torch.prod(hw, 1)
        # area_i = torch.prod(br - tl, 1) * en
        # 并集的面积
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            # 最小外接矩形的面积
            area_c = torch.prod(c_br - c_tl, 1)
            # area_c.clamp(1e-16)意义为将area_c的值下限设为1e-16，防止报错
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            # giou.clamp(min=-1.0, max=1.0)将giou值域限制为（-1，1），实际上giou的值也就是这个值
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        # 尝试加入diou，ciou
        elif self.loss_type == 'diou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            # 最大外界矩形对角线长度c^2
            w_c = (c_br - c_tl)[:, 0]
            h_c = (c_br - c_tl)[:, 1]
            c = w_c ** 2 + h_c ** 2
            # 中心点距离平方d^2
            w_d = (pred[:, :2] - target[:, :2])[:, 0]
            h_d = (pred[:, :2] - target[:, :2])[:, 1]
            d = w_d ** 2 + h_d ** 2
            # 求diou
            diou = iou - d/c
            loss = 1 - diou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == 'ciou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )

            # 最大外界矩形对角线长度c^2
            w_c = (c_br - c_tl)[:, 0]
            h_c = (c_br - c_tl)[:, 1]
            c = w_c ** 2 + h_c ** 2
            # 中心点距离平方d^2
            w_d = (pred[:, :2] - target[:, :2])[:, 0]
            h_d = (pred[:, :2] - target[:, :2])[:, 1]
            d = w_d ** 2 + h_d ** 2
            # 求diou

            diou = iou - d / c

            w_gt = target[:, 2]
            h_gt = target[:, 3]
            w = pred[:, 2]
            h = pred[:, 3]

            with torch.no_grad():
                arctan = torch.atan(w_gt / h_gt) - torch.atan(w / h)
                v = (4 / (math.pi ** 2)) * torch.pow(arctan, 2)
                s = 1 - iou
                alpha = v / (s + v)

            ciou = diou - alpha * v
            loss = 1-ciou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'siou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            cw = (c_br - c_tl)[:, 0]    #凸（最小的封闭框）宽度
            ch = (c_br - c_tl)[:, 1]
            #h中心点偏移量
            s_cw = target[:, 0] - pred[:, 0]
            s_ch = target[:, 1] - pred[:, 1]

            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5) + 1e-16

            sin_alpha = torch.abs(s_cw) / sigma
            sin_beta = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha0 = torch.where(sin_alpha > threshold, sin_beta, sin_alpha)
            # angle_cost = 1- 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - math.pi / 4),2)
            angle_cost = torch.cos(torch.arcsin(sin_alpha0)*2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            delta_x = 1-torch.exp(gamma*rho_x)
            delta_y = 1 - torch.exp(gamma * rho_y)
            distance_cost = delta_x+delta_y
            w_gt = target[:,2]
            h_gt = target[:, 3]
            w_pred = pred[:, 2]
            h_pred = pred[:, 3]
            W_w = torch.abs(w_pred-w_gt) / (torch.max(w_pred,w_gt))
            W_h = torch.abs(h_pred - h_gt) / torch.max(h_pred, h_gt)
            theta = 4
            shape_cost = torch.pow(1 - torch.exp(-1 * W_w), theta) + torch.pow(1 - torch.exp(-1 * W_h), theta)
            siou = iou - (distance_cost+shape_cost) * 0.5
            loss = 1 - siou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

