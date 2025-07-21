#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

#---数据增强---  lm
The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
# from yolox.utils import visualize,plot_rbbox


def adjust_box_anns1(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0:1] = np.clip(bbox[:, 0:1] * scale_ratio + padw, 0, w_max)
    bbox[:, 1:2] = np.clip(bbox[:, 1:2] * scale_ratio + padh, 0, h_max)
    bbox[:, 2:3] = np.clip(bbox[:, 2:3] * scale_ratio, 0, w_max)
    bbox[:, 3:4] = np.clip(bbox[:, 3:4] * scale_ratio, 0, h_max)
    return bbox

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates

def box_candidates2(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2], box1[3]
    w2, h2 = box2[2], box2[3]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


#------------候选框------------------
def box_candidates1(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    # w1, h1 = box1[2] - box1[0], box1[3] - box1[1]  #--lm
    # w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    a=[0,0,0,0]
    HH, WW = (640-2) , (640-2)

    for iii in range(0, 4):
        if box1[iii:iii + 1, 0] >= 7 and box1[iii:iii + 1, 1] >= 21.45:
            if box1[iii:iii + 1, 0] + box1[iii:iii + 1, 2] / 2 < 1 or box1[iii:iii + 1, 0] + box1[iii:iii + 1,2] / 2 > 2*HH or box1[iii:iii + 1, 1] + box1[iii:iii + 1,3] / 2 > 2*WW or box1[iii:iii + 1, 1] + box1[iii:iii + 1,3] / 2 < 1:
                a[iii] = 0
            else:
                a[iii] = 1
        else:
            a[iii] = 0

    return (
            ( a[0] > 0 )
            , (a[1] > 0)
            , (a[2] >0)
            , (a[3] >0 )
            )


def random_perspective(
    img, targets=(), degrees=0, translate=0.0, scale=0.1, shear=0.0, perspective=0.0, border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1]) #1.0
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
    T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:   #不满足
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points   #弯曲的点
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes   创建新bbox
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T   #（8，4）

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates  筛选候选框
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    # Transform label coordinates
    # n = len(targets)      #（4，6）
    # if n:
    #     # warp points
    #     xy = np.ones((n * 4, 3))   #改(n * 4, 3)
    #
    #     # filter candidates  筛选候选框
    #     i = box_candidates1(box1=targets[:, :4], box2=xy.T) #[:, :4]
    #     # i = np.array(i)
    #     targets = targets[i]
    #     # targets[:, :4] = xy[i]   #   改[:, :4]
    #
    #     # pts1, pts2, pts3, pts4 = plot_rbbox(targets)
    #     # points = np.array(
    #     #     [[pts1[0][0], pts1[0][1]], [pts2[0][0], pts2[0][1]], [pts3[0][0], pts3[0][1]], [pts4[0][0], pts4[0][1]]],
    #     #     np.int32)  # 多边形的顶点坐标
    #     # cv2.polylines(img, [points], True, 255, 2)  # 画任意多边形
    #     #
    #     # plt.imshow(img)
    #     # plt.show()

    return img, targets


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0:3:2] = width - boxes[:, 2::-2]   #改  --lm 因为加了一个角度进去，故  [:, 0::2]
    return image, boxes


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])     #理论要求输入/实际输入
    # resized_img图片 即resize成输入图片
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    #上面主要是将原图片按最小缩放比缩放后的空白区域填上灰度

    image = padded_img

    image = image.astype(np.float32)
    image = image[:, :, ::-1]


    image /= 255.0
    if mean is not None:
        image -= mean
    if std is not None:
        image /= std

    # plt.title("data ", fontsize=15)
    # plt.imshow(image)
    # plt.show()
    image = image.transpose(swap)   # 跳整通道位置
    image = np.ascontiguousarray(image, dtype=np.float32) #变为连续储存空间，有利于加速计算
    return image, r


def resize_lm(image, input_size):

    # img = np.array(image)
    img = image
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])     #理论要求输入/实际输入
    # resized_img图片 即resize成输入图片
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    #上面主要是将原图片按最小缩放比缩放后的空白区域填上灰度
    image = resized_img
    # image = image.astype(np.float32)
    return image, r


class TrainTransform:    #从G:\Github_download\yolo\Week2_yolox_lmxz\YOLOX-main\exps\example\yolox_voc\yolox_voc_s.py 73行跳转过来
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=50):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels   #只运行到这里就跳转回去了   ----lm

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :5].copy()   #改  --lm [:, :4]
        labels = targets[:, 5].copy() #改  --lm [:, :4]
        if targets.shape[1] > 6:  #改  --lm shape[1] > 5
            mixup = True
            ratios = targets[:, -1].copy()
            ratios_o = targets[:, -1].copy()
        else:
            mixup = False
            ratios = None
            ratios_o = None
        lshape = 7 if mixup else 6   #改  --lm
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, lshape), dtype=np.float32)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :5]   #包含角度
        labels_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]      #---输入就是xywh 故屏蔽226-233    --lm
        # b_x_o = (boxes_o[:, 2] + boxes_o[:, 0]) * 0.5
        # b_y_o = (boxes_o[:, 3] + boxes_o[:, 1]) * 0.5
        # b_w_o = (boxes_o[:, 2] - boxes_o[:, 0]) * 1.0
        # b_h_o = (boxes_o[:, 3] - boxes_o[:, 1]) * 1.0
        # boxes_o[:, 0] = b_x_o
        # boxes_o[:, 1] = b_y_o
        # boxes_o[:, 2] = b_w_o
        # boxes_o[:, 3] = b_h_o

        image_t = _distort(image)
        image_t, boxes = _mirror(image_t, boxes)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)
        boxes = boxes.copy()
        # boxes [xyxy] 2 [cx,cy,w,h]   #改     --lm 数据格式为xywh
        # b_x = (boxes[:, 2] + boxes[:, 0]) * 0.5
        # b_y = (boxes[:, 3] + boxes[:, 1]) * 0.5
        # b_w = (boxes[:, 2] - boxes[:, 0]) * 1.0
        # b_h = (boxes[:, 3] - boxes[:, 1]) * 1.0
        # boxes[:, 0] = b_x
        # boxes[:, 1] = b_y
        # boxes[:, 2] = b_w
        # boxes[:, 3] = b_h

        boxes[:, 0:4] *= r_      #  前四个参数除比值  --lm

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8  #gai  8    #gai
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        if mixup:
            ratios_t = ratios[mask_b].copy()

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o[:, 0:4] *= r_o    #--lm    boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            ratios_t = ratios_o

        labels_t = np.expand_dims(labels_t, 1)
        if mixup:
            ratios_t = np.expand_dims(ratios_t, 1)
            targets_t = np.hstack((labels_t, boxes_t, ratios_t))
        else:
            targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, lshape))   #填充标签
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        return torch.from_numpy(img), torch.zeros(1, 5)
