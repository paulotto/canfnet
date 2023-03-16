#!/usr/bin/env python3
"""
Functions for predicting the normal force and its distribution acting on the gel of a visuotactile sensor.
"""
__author__ = 'Paul-Otto Müller'
__date__ = '23.10.2022'

import torch
import logging
import numpy as np
import collections
import torchvision.transforms as T
from pathlib import Path
from torch import Tensor
from typing import Optional, Tuple

from .unet import UNet
from canfnet.utils.utils import PrintColors

RESIZE: Tuple[int, int] = (0, 0)
FILTER_SIZE: int = 5

# GelSight Mini.
'''
NORM_IMG: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.4907543957, 0.4985137582, 0.4685586393]),
                                             Tensor([0.0307641067, 0.0246135872, 0.0398434214]))    # (mean, std).
NORM_DIS: Optional[Tuple[Tensor, Tensor]] = (Tensor([-6.0596092226e-05]), Tensor([0.0002053244]))   # (mean, std).
'''
# DIGIT.
NORM_IMG: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.5024564266, 0.4860377908, 0.5020657778]),
                                             Tensor([0.0415902548, 0.0462468602, 0.0575232506]))    # (mean, std).
NORM_DIS: Optional[Tuple[Tensor, Tensor]] = (Tensor([-0.0001196197]), Tensor([0.0003911761]))       # (mean, std).

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

force_deque = collections.deque(maxlen=FILTER_SIZE)


def normalize(image: np.ndarray, norm: Tuple[Tensor, Tensor]) -> Tensor:
    """
    Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the
    range [0.0, 1.0] if the numpy.ndarray has dtype = np.uint8. Subsequently, the tensor is normalized with the given
    mean and standard deviation values.

    :param image: An image to be normalized.
    :param norm: The mean and std deviation for each channel.
    :return: The normalized image as Tensor.
    """
    normalize_ = T.Compose([T.ToTensor(), T.Normalize(norm[0], norm[1])])
    return normalize_(image)


def load_unet(model_path: Path, torch_device: torch.device = 'cuda') -> UNet:
    """
    Loads and returns a trained U-Net PyTorch model.

    :param model_path: The model name including the path.
    :param torch_device: The device used for inference ('cuda': GPU, 'cpu': CPU).
    :return: The loaded UNet model.
    """
    if torch_device == 'cuda':
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unet: UNet = UNet(padding='same', padding_mode='zeros', resize_output=RESIZE)

    unet.load_state_dict(torch.load(str(model_path), map_location=torch_device)['state_dict'])
    unet = unet.to(device=torch_device)
    unet.eval()

    logging.info(f'{PrintColors.OKBLUE}Model loaded from {model_path}{PrintColors.ENDC}\n')
    logging.info(f'Using device {torch_device}\n')

    return unet


@torch.no_grad()
def predict(image: np.ndarray,
            unet: UNet,
            torch_device: torch.device = 'cuda',
            norm_img: Optional[Tuple[Tensor, Tensor]] = NORM_IMG,
            norm_dis: Optional[Tuple[Tensor, Tensor]] = NORM_DIS,
            force_filter: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts the normal force and its distribution acting on the gel from (normalized) visuotactile images.

    :param image: A raw visuotactile image.
    :param unet: A U-Net PyTorch model.
    :param torch_device: The device used for inference ('cuda': GPU, 'cpu': CPU).
    :param norm_img: The mean and std deviation for each channel to normalize the input image.
    :param norm_dis: The mean and std deviation to de-normalize the predicted normal force distribution.
    :param force_filter: True if the estimated normal force should be (median) filtered.
    :return: The predicted normal force and normal force distribution.
    """
    if torch_device == 'cuda':
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unet = unet.to(device=torch_device)
    unet.eval()
    inv_normalize_dis = T.Normalize(mean=[-m / s for m, s in zip(norm_dis[0], norm_dis[1])],
                                    std=[1 / s for s in norm_dis[1]]) if norm_dis is not None else lambda x: x

    # Normalize the input image.
    if norm_img is not None:
        image: Tensor = normalize(image, norm_img).to(device=torch_device).float()
    else:
        image: Tensor = torch.from_numpy(image).to(device=torch_device).float()

    image: Tensor = torch.unsqueeze(image, dim=0)

    # Estimate the normal force and its distribution and invert the normalization.
    f_dis_: Tensor = unet(image)
    f_dis_ = inv_normalize_dis(f_dis_)

    f_dis_: np.ndarray = f_dis_[0].permute(1, 2, 0).cpu().numpy()
    force_: np.ndarray = np.sum(f_dis_)

    # Filter the estimated force.
    global force_deque
    force_deque.append(force_)
    if force_filter:
        force_ = np.median(force_deque)

    return force_.astype(np.float32), f_dis_.astype(np.float32)
