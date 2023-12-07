#!/usr/bin/env python3
"""
Metric functions and classes.

Classes:
    - AverageMeter
    - VistacLossMeter

Functions:
    - epoch_log(epoch_loss: float, meter: VistacLossMeter) -> Union[Tuple[float, float, float], Tuple[float, float]]
    - iou(input_: Tensor, force_dis_lbl: Tensor, threshold: float = 6e-4) -> Tensor
"""
__author__ = 'Paul-Otto Müller'
__date__ = '13.09.2022'

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import List, Tuple, Union

from utils.loss import VistacLoss, VistacLossSep


class AverageMeter:
    """
    A class for tracking the average loss of one epoch.
    """
    def __init__(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VistacLossMeter:
    """
    Tracks the metrics for either a U-Net with one or two outputs. That is, the MSE and IFL for one output and the MSE,
    IoU and Focal loss for two outputs.
    """
    def __init__(self, unet_sep: bool = False) -> None:
        self.unet_sep: bool = unet_sep
        self.mse_loss: List[float] = []
        self.if_loss: List[float] = []
        self.iou_loss: List[float] = []
        self.focal_loss: List[float] = []

    def update(self, vistac_loss: Union[VistacLoss, nn.Module]) -> None:
        if isinstance(vistac_loss, VistacLoss):
            self.mse_loss.append(vistac_loss.mse_actual.item())
            self.if_loss.append(vistac_loss.ifl_actual.item())
        if isinstance(vistac_loss, VistacLossSep):
            self.mse_loss.append(vistac_loss.mse_actual.item())
            self.iou_loss.append(vistac_loss.iou_actual.item())
            self.focal_loss.append(vistac_loss.focal_actual.item())

    def get_metrics(self) -> Union[Tuple[float, float, float], Tuple[float, float]]:
        if self.unet_sep:
            return float(np.nanmean(self.mse_loss)), \
                   float(np.nanmean(self.iou_loss)), float(np.nanmean(self.focal_loss))
        else:
            return float(np.nanmean(self.mse_loss)), float(np.nanmean(self.if_loss))


def epoch_log(epoch_loss: float, meter: VistacLossMeter) -> Union[Tuple[float, float, float], Tuple[float, float]]:
    """
    Logging the metrics at the end of an epoch.

    :param epoch_loss: The loss from one epoch.
    :param meter: A VistacLossMeter that tracked all metrics.
    :return: Either (total loss, MSE, IoU, Focal) if a U-Net with two outputs is used for training or
    (total loss, MSE, IFL) if the U-Net with one output is used.
    """
    loss = meter.get_metrics()

    if meter.unet_sep:
        print("Loss: %0.4f | mse: %0.4f | iou: %0.4f | focal: %0.4f" % (epoch_loss, loss[0], loss[1], loss[2]))
    else:
        print("Loss: %0.4f | mse: %0.4f | ifl: %0.4f" % (epoch_loss, loss[0], loss[1]))

    return loss


def iou(input_: Tensor, force_dis_lbl: Tensor, threshold: float = 6e-4) -> Tensor:
    """
    Computes the Intersection over Union (IoU) metric for an estimated force distribution by thresholding and converting
    both the estimated and true force distributions to a binary mask.

    :param input_: The estimated force distribution, already de-normalized.
    :param force_dis_lbl: The ground truth force distribution.
    :param threshold: The value which will be used to create a binary mask from the input. Basically, at which
    force per pixel value the value should be not treated as zero anymore.
    :return: The IoU value.
    """
    smooth = 1e-6
    input_ = (torch.abs(input_) > threshold).int()
    force_dis_lbl = (torch.abs(force_dis_lbl) > 0.0).int()

    intersection = (input_ & force_dis_lbl).float().sum((2, 3))
    union = (input_ | force_dis_lbl).float().sum((2, 3))

    iou_ = (intersection + smooth) / (union + smooth)

    return iou_.mean()
