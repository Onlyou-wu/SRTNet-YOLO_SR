# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math

import torch
import torch.nn as nn
from utils.ops import *


from models.common import *

from .talv8 import dist2bbox, dist2rbox, make_anchors
# from .block import DFL

__all__ = "Detectv8", "OBB"


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Detectv8(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers 3
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4    # number of outputs per anchor   #66
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(    # 用于预测每个锚点的位置信息   #4--》5
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 5 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


    def inference(self, x):
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def forward_feat(self, x, cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        y = self.forward_feat(x, self.cv2, self.cv3)
        
        if self.training:
            return y

        return self.inference(y)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):    # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        if self.export:
            return dist2bbox(bboxes, anchors, xywh=False, dim=1)
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)

class OBB(Detectv8):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters
        self.detect = Detectv8.forward

#改  将去掉单独的角度回归分支
        # c4 = max(ch[0] // 4, self.ne)
        # self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        #        连接并返回预测的边界框和类别概率
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits

        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        #注意： 将 `angle` 设置为属性，以便 `decode_bboxes` 可以使用它
        # angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]   # exp4  0.605
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]   # exp5  0.442
        # print(angle)
        angle = (2*angle.sigmoid()-1) * math.pi   # [-pi, pi] # exp5  0.741  +100epoch   0.758  +100epoch+使用300的数据集 0.694，检测效果不好，框都是正的
        # 不使用激活函数 0.668  exp3


        if not self.training:
            self.angle = angle
        x = self.detect(self, x)    #得到上面数据
        if self.training:
            return x, angle
        #self.export为flase 返回return torch.cat([x, angle], 1)
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))


    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        #解码边界框
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

import copy
class v10Detect(Detectv8):
    max_det = 300

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)), \
                                               nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)), \
                                               nn.Conv2d(c3, self.nc, 1)) for i, x in enumerate(ch))

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.export:
            one2many = super().forward(x)

        if not self.training:
            one2one = self.inference(one2one)
            if not self.export:
                return {"one2many": one2many, "one2one": one2one}
            else:
                assert (self.max_det != -1)
                from utils import ops
                boxes, scores, labels = ops.v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)
        else:
            return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        super().bias_init()
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

