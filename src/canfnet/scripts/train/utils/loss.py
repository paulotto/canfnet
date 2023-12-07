#!/usr/bin/env python3
"""
Loss functions.

Reference:
    - [https://github.com/amaarora/amaarora.github.io/blob/master/nbs/Training.ipynb]

Classes:
    -

Functions:
    -
"""
__author__ = 'Paul-Otto Müller'
__date__ = '13.09.2022'

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple


def dice_loss(input_, target):
    input_ = torch.sigmoid(input_)
    smooth = 1.0
    iflat = input_.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def iou_seg(input_: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    smooth = 1e-6
    input_ = (torch.sigmoid(input_) > threshold).int()

    intersection = (input_ & labels).float().sum((2, 3))
    union = (input_ | labels).float().sum((2, 3))

    iou_ = (intersection + smooth) / (union + smooth)

    # thresholded = torch.clamp(20 * (iou_ - 0.5), 0, 10).ceil() / 10

    return iou_.mean()


def integrated_force_loss(input_: Tensor,
                          force_dis: Tensor,
                          area: Tensor,
                          force: Tensor,
                          mode: str = 'pixel',
                          norm_dis: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
    """
    TODO
    """
    if norm_dis is not None:
        inv_normalize_dis = T.Normalize(mean=[-m / s for m, s in zip(norm_dis[0], norm_dis[1])],
                                        std=[1 / s for s in norm_dis[1]])
        input_ = inv_normalize_dis(input_)

    if mode == 'mm':
        nr_pixels: Tensor = Tensor([torch.count_nonzero(force_dis[i, :, :, :]) for i in range(force_dis.shape[0])])
        nr_pixels = nr_pixels.to(device=input_.device)
        pixel_area_l: List[Tensor] = list(area / nr_pixels)
        pixel_area_l = [t.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0) for t in pixel_area_l]
        force_values_l: List[Tensor] = [input_[i][input_[i] != 0.0] for i in range(input_.size()[0])]

        force_reconstructed = [torch.sum(f * px_area) for f, px_area in zip(force_values_l, pixel_area_l)]
        force_reconstructed = torch.stack(force_reconstructed)
    else:
        force_reconstructed = torch.sum(input_, (2, 3)).view(input_.size(0))

    return F.mse_loss(force_reconstructed, force, reduction='mean')


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.35, gamma: float = 2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha: Tensor = torch.tensor([alpha, 1 - alpha])
        self.gamma: float = gamma

    def forward(self, input_: Tensor, target: Tensor):
        self.alpha = self.alpha.to(device=input_.device)
        BCE_loss = F.binary_cross_entropy_with_logits(input_, target.float(), reduction='none')
        target = target.type(torch.int64)
        at = self.alpha.gather(0, target.view(-1)).view(target.shape)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class VistacLoss(nn.Module):
    """
    TODO
    """

    def __init__(self,
                 mse_scale: float = 1.0,
                 c_ifl: float = 0.0,
                 mode: str = 'pixel',
                 norm_dis: Optional[Tuple[Tensor, Tensor]] = (Tensor([-0.0001470775]), Tensor([0.0003999361]))) -> None:
        super().__init__()
        self.mse_scale: float = mse_scale
        self.c_ifl: float = c_ifl
        self.mode: str = mode
        self.norm_dis = norm_dis
        self.mse: nn.Module = nn.MSELoss(reduction='mean')
        self.areas: Tensor = torch.ones(1)
        self.forces: Tensor = torch.ones(1)
        self.mse_actual: Tensor = torch.zeros(1)
        self.ifl_actual: Tensor = torch.zeros(1)

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        self.mse_actual = self.mse(input_, target)
        self.ifl_actual = integrated_force_loss(input_, target, self.areas, self.forces,
                                                mode=self.mode, norm_dis=self.norm_dis)
        return self.mse_scale * self.mse_actual + self.c_ifl * self.ifl_actual


class VistacLossSep(nn.Module):
    def __init__(self, mse_scale: float = 1.0,
                 iou_scale: float = 1.0,
                 focal_params: Tuple[float, float, float] = (10.0, 0.35, 2.0)) -> None:
        super().__init__()
        self.mse_scale: float = mse_scale
        self.iou_scale: float = iou_scale
        self.focal_params: Tuple[float, float, float] = focal_params
        self.focal_loss: WeightedFocalLoss = WeightedFocalLoss(self.focal_params[1], self.focal_params[2])
        self.mse: nn.Module = nn.MSELoss(reduction='mean')
        self.forces: Tensor = torch.ones(1)
        self.mse_actual: Tensor = torch.zeros(1)
        self.iou_actual: Tensor = torch.zeros(1)
        self.focal_actual: Tensor = torch.zeros(1)

    def forward(self, input_: Tuple[Tensor, Tensor], target: Tuple[Tensor, Tensor]) -> Tensor:
        self.mse_actual = self.mse(input_[0], target[0])
        self.iou_actual = iou_seg(input_[1], target[1])
        self.focal_actual = self.focal_loss(input_[1], target[1])
        return self.mse_scale * self.mse_actual \
               + self.focal_params[0] * self.focal_actual - self.iou_scale * torch.log(self.iou_actual)
