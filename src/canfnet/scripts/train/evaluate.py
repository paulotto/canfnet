#!/usr/bin/env python3
"""
Evaluating a trained U-Net.
"""
__author__ = 'Paul-Otto Müller'
__date__ = '31.10.2022'

import argparse
import logging
from pathlib import Path
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Dict, Any
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from digit_interface import Digit

import utils.metric
import utils.util as util
from unet.unet import UNet, UNetSep
from utils.loss import VistacLoss, VistacLossSep
from utils.dataset import VistacDataSet
from utils.loss import iou_seg

# plt.style.use('ggplot')

PROJECT_DIR: Path = Path(__file__).parent.resolve()
MODELS_DIR: Path = Path(PROJECT_DIR, 'models')
DATA_DIR: Path = Path(PROJECT_DIR, 'training_data')
BATCH_SIZE: int = 10
RESIZE: Tuple[int, int] = (0, 0)
MSE_SCALE: float = 1.0
C_IFL: float = 0.05
IOU_SCALE: float = 1.0
FOCAL_PARAMS: Tuple[float, float, float] = (10.0, 0.35, 2.0)
UNET_SEP: bool = False

# Gelsight Mini.
NORM_IMG: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.4907543957, 0.4985137582, 0.4685586393]),
                                             Tensor([0.0307641067, 0.0246135872, 0.0398434214]))    # (mean, std).
NORM_LBL: Optional[Tuple[Tensor, Tensor]] = (Tensor([-6.0596092226e-05]), Tensor([0.0002053244]))   # (mean, std).
# Digit.
'''
NORM_IMG: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.5024564266, 0.4860377908, 0.5020657778]),
                                             Tensor([0.0415902548, 0.0462468602, 0.0575232506]))  # (mean, std).
NORM_LBL: Optional[Tuple[Tensor, Tensor]] = (Tensor([-0.0001196197]), Tensor([0.0003911761]))     # (mean, std).
'''


def identity(x) -> Any:
    """
    Returns the input.

    :param x: Anything.
    :return: The input.
    """
    return x


def batch_to_device(batch: Dict[str, Tensor], dev: torch.device) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Pushes a batch from a 'VistacDataSet' to the given device and returns the contents of the batch individually
    as tuple.

    :param batch: A batch from a 'VistacDataSet'.
    :param dev: The device to which the data is to be pushed.
    :return: images, force distribution label, measured indenter areas, measured force.
    """
    images, f_dis, areas, f = batch.values()
    return images.to(dev), f_dis.to(dev), areas.to(dev), f.to(dev)


@torch.no_grad()
def forward(net: Union[UNet, UNetSep],
            batch: Dict[str, Tensor],
            vistac_loss: Union[VistacLoss, VistacLossSep],
            dev: torch.device,
            mode: str = 'pixel') -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes the total and corresponding individual losses for a U-Net with one or two outputs for one run through
    the forward pass of the network and returns them.

    :param net: A U-Net.
    :param batch: A batch of data from a 'VistacDataSet'.
    :param vistac_loss: Either a 'VistacLoss' or 'VistacLossSep'.
    :param dev: The device to which the data is to be pushed.
    :param mode: Either 'pixel' or 'N/mm'.
    :return: One-output U-Net: total loss, MSE, IF, IoU, (MAE of force).
    Two-output U-Net: total loss, MSE, IoU, Focal loss, MAE.
    """
    inv_normalize_dis = T.Normalize(mean=[-m / s for m, s in zip(NORM_DIS[0], NORM_DIS[1])],
                                    std=[1 / s for s in NORM_DIS[1]]) if NORM_DIS is not None else identity

    images, f_dis, areas, forces = batch_to_device(batch, dev)

    if isinstance(net, UNetSep):
        f_dis[f_dis != 0] = 1  # Convert to binary mask.
        outputs = net(images)
        forces = forces.view(-1, 1)
        f_pred_norm = outputs[0] * (net.max_force.to(dev) - net.min_force.to(dev)) + net.min_force.to(dev)
        outputs = (f_pred_norm, outputs[1])

        total_loss = vistac_loss(outputs, (forces, f_dis.int()))
        mae = F.l1_loss(f_pred_norm, forces)
        return total_loss, vistac_loss.mse_actual, vistac_loss.iou_actual, vistac_loss.focal_actual, mae
    else:
        outputs = net(images)
        vistac_loss.areas = areas
        vistac_loss.forces = forces

        total_loss = vistac_loss(outputs, f_dis)
        iou_loss = utils.metric.iou(inv_normalize_dis(outputs), inv_normalize_dis(f_dis), threshold=6e-4)

        mae = torch.tensor([0.0], device=dev)
        if mode == 'pixel':
            force_reconstructed = torch.sum(inv_normalize_dis(outputs), (2, 3)).view(outputs.size(0))
            mae = F.l1_loss(force_reconstructed, forces, reduction='mean')

        return total_loss, vistac_loss.mse_actual, vistac_loss.ifl_actual, iou_loss, mae


@torch.no_grad()
def evaluate(net: Union[UNet, UNetSep],
             dataset: VistacDataSet,
             dev: torch.device = 'cuda',
             mode: str = 'pixel') -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluates a U-Net by computing the mean total loss and its corresponding components. According to the U-Net
    structure, some returned values might be zero.

    :param net: A U-Net.
    :param dataset: A 'VistacDataSet'.
    :param dev: The device to which the data is to be pushed.
    :param mode: Either 'pixel' or 'N/mm'.
    :return: Total loss used for training, MSE, IoU, IF, Focal loss.
    """
    if dev == 'cuda':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=dev)
    net.eval()

    loader_: DataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    nr_batches: int = len(loader_) if len(loader_) != 0 else 1
    vistac_loss: VistacLoss = VistacLoss(mse_scale=MSE_SCALE, c_ifl=C_IFL, mode=mode, norm_dis=NORM_DIS)
    vistac_loss_sep: VistacLossSep = VistacLossSep(mse_scale=MSE_SCALE, iou_scale=IOU_SCALE, focal_params=FOCAL_PARAMS)

    total_loss: Tensor = torch.tensor([0.0], device=dev)
    mse_loss: Tensor = torch.tensor([0.0], device=dev)
    iou_loss: Tensor = torch.tensor([0.0], device=dev)
    if_loss: Tensor = torch.tensor([0.0], device=dev)
    focal_loss: Tensor = torch.tensor([0.0], device=dev)
    mae_loss: Tensor = torch.tensor([0.0], device=dev)

    for batch in tqdm(loader_, desc='Evaluating given dataset', unit='batch'):
        if_loss_i, focal_loss_i = torch.tensor([0.0], device=dev), torch.tensor([0.0], device=dev)

        if isinstance(net, UNetSep):
            total_loss_i, mse_loss_i, iou_loss_i, focal_loss_i, mae_loss_i = forward(net, batch, vistac_loss_sep, dev, mode)
        else:
            total_loss_i, mse_loss_i, if_loss_i, iou_loss_i, mae_loss_i = forward(net, batch, vistac_loss, dev, mode)

        total_loss += total_loss_i.sum(0)
        mse_loss += mse_loss_i.sum(0)
        iou_loss += iou_loss_i.to(dev)
        if_loss += if_loss_i.sum(0)
        focal_loss += focal_loss_i.sum(0)
        mae_loss += mae_loss_i.sum(0)

    total_loss /= nr_batches
    mse_loss /= nr_batches
    iou_loss /= nr_batches
    if_loss /= nr_batches
    focal_loss /= nr_batches
    mae_loss /= nr_batches

    logging.info(f'''Results:
                Total (used for training): {total_loss.cpu().item():3.4f}
                MSE:                       {mse_loss.cpu().item():3.4f}
                RMSE:                      {torch.sqrt(mse_loss).cpu().item():3.4f}
                IoU:                       {iou_loss.cpu().item():3.4f}
                IF:                        {if_loss.cpu().item():3.4f}
                RIF:                       {torch.sqrt(if_loss).cpu().item():3.4f}
                Focal:                     {focal_loss.cpu().item():3.4f}
                MAE (force):               {mae_loss.cpu().item():3.4f}\n''')
    return total_loss, mse_loss, iou_loss, if_loss, focal_loss


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluation of a trained U-Net')
    parser.add_argument('--model', '-m', default=MODELS_DIR, metavar='FILE',
                        help='Specify the file in which the model is stored', required=True)
    parser.add_argument('--data', '-d', default=DATA_DIR, metavar='DIR',
                        help='Specify the directory in which the data is stored', required=False)
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='val', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--separate', '-sep', action='store_true', default=False,
                        help='Use the U-Net with two separate outputs.')
    parser.add_argument('--resize_output', '-r', dest='resize', default=RESIZE, metavar='(H, W)',
                        help='Resize the output to (H, W)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    RESIZE = args.resize
    UNET_SEP = args.separate
    MODELS_DIR = args.model
    DATA_DIR = args.data
    BATCH_SIZE = args.batch_size

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if UNET_SEP:
        unet: UNetSep = UNetSep(padding='same', padding_mode='zeros', resize_output=RESIZE)
    else:
        unet: UNet = UNet(padding='same', padding_mode='zeros', resize_output=RESIZE)

    unet.to(device=device)
    unet.load_state_dict(torch.load(args.model, map_location=device)['state_dict'])

    logging.info(f'{util.PrintColors.OKBLUE}Model loaded from {args.model}{util.PrintColors.ENDC}')
    logging.info(f'Using device {device}\n')

    if UNET_SEP:
        vistac_set: VistacDataSet = VistacDataSet(str(DATA_DIR), norm_img=NORM_IMG, norm_lbl=None, augment=False)
    else:
        vistac_set: VistacDataSet = VistacDataSet(str(DATA_DIR), norm_img=NORM_IMG, norm_lbl=NORM_DIS, augment=False)
    logging.info(f'Dataset size: {len(vistac_set)}')
    logging.info(f'Batch size: {BATCH_SIZE}\n')

    evaluate(unet, vistac_set, device, mode='pixel')
