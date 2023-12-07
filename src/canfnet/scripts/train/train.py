#!/usr/bin/env python3
"""
Training of the U-Net.

Reference:
    - [https://github.com/amaarora/amaarora.github.io/blob/master/nbs/Training.ipynb]
    - [https://github.com/milesial/Pytorch-UNet]
"""
__author__ = 'Paul-Otto Müller'
__date__ = '12.09.2022'

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision as tv
import torchvision.transforms as T
import neptune.new as neptune
import matplotlib.pyplot as plt
from torch import optim, Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
from datetime import datetime

import utils.util as util
import utils.metric
import utils.dataset
import utils.loss
from unet.unet import UNet, UNetSep
from utils.loss import VistacLoss, VistacLossSep


PROJECT_DIR: Path = Path(__file__).parent.resolve()
MODELS_DIR: Path = Path(PROJECT_DIR, 'models')
TRAIN_DIR: Path = Path(PROJECT_DIR, 'training_data')
DEVICE: str = 'cpu'
FOLDS: Optional[int] = None
EPOCHS: int = 5
TRAIN_BATCH_SIZE: int = 1
VAL_BATCH_SIZE: int = 1
LEARNING_RATE: float = 1e-3
OUTPUT_CHANNELS: int = 1
VAL_PERCENT: float = 0.2
OPTIMIZER: str = 'Adam'
SCHEDULER: str = 'ReduceLROnPlateau'
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
NUM_WORKERS: int = 1    # Can lead to errors if greater than 1 (https://github.com/numpy/numpy/issues/18124).
PIN_MEMORY: bool = True
AUGMENT: bool = True
MMAP_MODE: Optional[str] = 'r'
CUDNN_BENCHMARK: bool = True
AMP: bool = False
DEBUG: bool = False
NEPTUNE: bool = False

UNET_SEP: bool = False
ENC_CHS: Tuple = (3, 32, 64, 128, 256)
DEC_CHS: Tuple = (256, 128, 64, 32)
OUT_CHS: int = 1
PADDING: str = 'same'
PADDING_MODE: str = 'zeros'
RESIZE_OUTPUT: Tuple[int, int] = (0, 0)
DROPOUT: float = 0.2
SE_BLOCK_R: int = 16
FORCE_NEURONS: Tuple[int, int, int] = (16, 8, 1)

if UNET_SEP:
    NORM_LBL = None
    CRITERION: nn.Module = VistacLossSep(mse_scale=1.0, iou_scale=1.0)
else:
    CRITERION: nn.Module = VistacLoss(mse_scale=1.0, c_ifl=0.01, mode='pixel', norm_dis=NORM_LBL)


class UNetTrainer(object):
    """
    UNetTrainer.
    """
    def __init__(self, net: nn.Module,
                 dev: torch.device,
                 optimizer_: torch.optim,
                 scheduler_: torch.optim.lr_scheduler,
                 loss_fn: nn.Module,
                 norm_img: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 norm_lbl: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 k: Optional[int] = 10,
                 augment: bool = False,
                 mmap_mode: Optional[str] = 'r') -> None:
        self.net: nn.Module = net.to(device=dev)
        self.device: torch.device = dev
        self.optimizer: torch.optim = optimizer_
        self.scheduler: torch.optim.lr_scheduler = scheduler_
        self.loss_fn: nn.Module = loss_fn.to(device=dev)
        self.norm_img = norm_img
        self.norm_lbl = norm_lbl
        self.augment: bool = augment
        self.mmap_mode: Optional[str] = mmap_mode
        self._grad_scaler = torch.cuda.amp.GradScaler(enabled=AMP)

        self.train_dir: Path = TRAIN_DIR
        self.epochs: int = EPOCHS
        self.batch_size: Dict[str, int] = {'train': TRAIN_BATCH_SIZE, 'val': VAL_BATCH_SIZE}
        self.lr: float = LEARNING_RATE
        self.out_chs: int = OUTPUT_CHANNELS
        self.num_workers: int = NUM_WORKERS
        self._val_percent: float = VAL_PERCENT
        self.k: Optional[int] = k
        if k is not None and k != 0 and k != 1:
            self._kfold: KFold = KFold(n_splits=k, shuffle=True, random_state=42)
        else:
            self._kfold = None

        self.phases: List[str] = ["train", "val"]
        self.accumulation_steps: int = 1
        self.best_loss = float("inf")

        train_loader, val_loader = self.create_data_loaders()
        self.data_loader: Dict[str, List[DataLoader]] = {'train': train_loader, 'val': val_loader}

        self.losses: Dict[str, List] = {phase: [] for phase in self.phases}
        self.integrated_force_loss: Dict[str, List] = {phase: [] for phase in self.phases}
        self.mse_loss: Dict[str, List] = {phase: [] for phase in self.phases}
        self.iou: Dict[str, List] = {phase: [] for phase in self.phases}

        cudnn.benchmark = CUDNN_BENCHMARK

    def forward(self, images: Tensor, f_dis: Tensor,  areas: Tensor, forces: Tensor) -> Tuple[Tensor, Tensor]:
        images = images.to(self.device)
        f_dis = f_dis.to(self.device)
        areas = areas.to(self.device)
        forces = forces.to(self.device)

        if isinstance(self.net, UNetSep):
            f_dis[f_dis != 0] = 1   # Convert to binary mask.
            outputs = self.net(images)
            forces = forces.view(-1, 1)
            f_pred_norm = outputs[0] * (self.net.max_force.to(self.device) - self.net.min_force.to(
                self.device)) + self.net.min_force.to(self.device)
            outputs = (f_pred_norm, outputs[1])
            loss = self.loss_fn(outputs, (forces, f_dis.int()))
        else:
            outputs = self.net(images)
            self.loss_fn.areas = areas
            self.loss_fn.forces = forces
            loss = self.loss_fn(outputs, f_dis)

        return loss, outputs

    def iterate(self, epoch: int, phase: str, data_loader: DataLoader) -> Tuple[float, Tuple]:
        start_t = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print(f"\nEpoch: {epoch} | phase: {phase} | date: {start_t}")

        batch_size = self.batch_size[phase]
        # data_loader = self.data_loader[phase]
        run_loss = utils.metric.AverageMeter()
        vistac_loss_meter = utils.metric.VistacLossMeter(unet_sep=UNET_SEP)
        total_batches = len(data_loader)
        self.net.train(phase == 'train')
        tk0 = tqdm(data_loader, desc=f"Epoch: {epoch}", total=total_batches, unit='batch')
        self.optimizer.zero_grad()

        for itr, batch in enumerate(tk0):
            images, f_dis, areas, f = batch.values()
            loss, _ = self.forward(images, f_dis, areas, f)

            if phase == 'train':
                with torch.set_grad_enabled(True):
                    # loss.backward()
                    self._grad_scaler.scale(loss).backward()
                    if (itr + 1) % self.accumulation_steps == 0:
                        # self.optimizer.step()
                        self._grad_scaler.step(self.optimizer)
                        self.optimizer.zero_grad()
                        self._grad_scaler.update()

            run_loss.update(loss.item(), batch_size)
            # outputs = outputs.detach().cpu()
            vistac_loss_meter.update(self.loss_fn)
            tk0.set_postfix(loss=run_loss.avg, learning_rate=self.optimizer.param_groups[0]['lr'])

        epoch_loss = run_loss.avg
        self.losses[phase].append(epoch_loss)
        loss = ()
        if isinstance(self.loss_fn, VistacLoss):
            loss = utils.metric.epoch_log(epoch_loss, vistac_loss_meter)
            self.mse_loss[phase].append(loss[0])
            self.integrated_force_loss[phase].append(loss[1])
        if isinstance(self.loss_fn, VistacLossSep):
            loss = utils.metric.epoch_log(epoch_loss, vistac_loss_meter)
            self.mse_loss[phase].append(loss[0])
            self.iou[phase].append(loss[1])

        if self.device != 'cpu':
            torch.cuda.empty_cache()

        return epoch_loss, loss

    def start(self) -> None:
        self._grad_scaler.__init__(enabled=AMP)

        # Initialize logging.
        neptune_run = self.init_neptune() if NEPTUNE else None

        val_len = len(self.data_loader['val'][0].dataset)
        logging.info(f'''Starting training:
                    Folds:            {FOLDS}
                    Epochs:           {EPOCHS}
                    Train batch size: {TRAIN_BATCH_SIZE}
                    Val batch size:   {VAL_BATCH_SIZE}
                    Learning rate:    {LEARNING_RATE}
                    Training size:    {len(self.data_loader['train'][0].dataset)}
                    Validation size:  {val_len // self.k if self.k is not None else val_len}
                    Device:           {device.type} \n''')

        start_time_ = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        for fold, (train_loader, val_loader) in enumerate(zip(self.data_loader['train'], self.data_loader['val'])):
            start_t = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            print(f"\n{util.PrintColors.OKBLUE}Fold: {fold + 1} | date: {start_t}{util.PrintColors.ENDC}")

            for epoch in range(1, self.epochs + 1):
                train_loss, train_loss_split = self.iterate(epoch, 'train', train_loader)
                self.net.eval()     # Important for ONNX to turn the model to inference mode before saving.
                state_ = {
                    'fold': fold,
                    'epoch': epoch,
                    'best_loss': self.best_loss,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                val_loss, val_loss_split = self.iterate(epoch, 'val', val_loader)
                self.scheduler.step(val_loss)
                if val_loss < self.best_loss:
                    print(f"{util.PrintColors.OKBLUE}------------- New optimum found, saving state -------------"
                          f"{util.PrintColors.ENDC}")
                    state_['best_loss'] = self.best_loss = val_loss
                    model_name_ = 'model_' + start_time_ + '.pth'
                    torch.save(state_, Path(MODELS_DIR, model_name_))
                print()

                if neptune_run is not None:
                    loss_split_name: Tuple = ('ifl',) if isinstance(self.loss_fn, VistacLoss) else ('iou', 'focal')
                    neptune_run["training/epoch/train_loss"].log(train_loss)
                    neptune_run["training/epoch/val_loss"].log(val_loss)
                    neptune_run["training/epoch/train_mse"].log(train_loss_split[0])
                    neptune_run["training/epoch/train_" + loss_split_name[0]].log(train_loss_split[1])
                    neptune_run["training/epoch/val_mse"].log(val_loss_split[0])
                    neptune_run["training/epoch/val_" + loss_split_name[0]].log(val_loss_split[1])
                    if len(loss_split_name) > 1:
                        neptune_run["training/epoch/train_" + loss_split_name[1]].log(train_loss_split[2])
                        neptune_run["training/epoch/val_" + loss_split_name[1]].log(train_loss_split[2])

        neptune_run.stop() if neptune_run is not None else 0

    def create_data_loaders(self) -> Tuple[List[DataLoader], List[DataLoader]]:
        train_loader_list_: List[DataLoader] = []
        val_loader_list_: List[DataLoader] = []

        # Create dataset.
        vistac_dataset = utils.dataset.VistacDataSet(str(self.train_dir), norm_img=self.norm_img,
                                                     norm_lbl=self.norm_lbl, augment=self.augment,
                                                     mmap_mode=self.mmap_mode)

        train_load_args = dict(batch_size=self.batch_size['train'], num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        val_load_args = dict(batch_size=self.batch_size['val'], drop_last=True,
                             num_workers=self.num_workers, pin_memory=PIN_MEMORY)

        if self._kfold is None:
            # Split into train / validation sets.
            n_val = int(len(vistac_dataset) * self._val_percent)
            n_train = len(vistac_dataset) - n_val
            train_set, val_set = random_split(vistac_dataset, [n_train, n_val],
                                              generator=torch.Generator().manual_seed(42))

            # Create data loaders.
            train_loader_list_.append(DataLoader(train_set, shuffle=True, **train_load_args))
            val_loader_list_.append(DataLoader(val_set, **val_load_args))
        else:
            for train_idx, val_idx in self._kfold.split(np.arange(len(vistac_dataset))):
                train_sampler_ = SubsetRandomSampler(train_idx)
                val_sampler_ = SubsetRandomSampler(val_idx)
                train_loader_ = DataLoader(vistac_dataset, sampler=train_sampler_, **train_load_args)
                val_loader_ = DataLoader(vistac_dataset, sampler=val_sampler_, **val_load_args)

                train_loader_list_.append(train_loader_)
                val_loader_list_.append(val_loader_)

        # Check.
        if DEBUG:
            data = next(iter(train_loader_list_[0]))
            images, f_dis, areas, f = data.values()
            print(f"area: {areas.size()}; image: {images.size()}; label: {f_dis.size()}; force: {f.size()}\n")
            img_grid = tv.utils.make_grid(images[:9], nrow=3, normalize=False)
            lbl_grid = tv.utils.make_grid(f_dis[:9], nrow=3, normalize=True)
            util.visualize_overlapped(img_grid, lbl_grid)
            util.visualize_overlapped(img_grid, None)

            if self.norm_img is not None and self.norm_lbl is not None:
                inv_normalize_img = T.Normalize(mean=[-m / s for m, s in zip(self.norm_img[0], self.norm_img[1])],
                                                std=[1 / s for s in self.norm_img[1]])
                inv_normalize_lbl = T.Normalize(mean=[-m / s for m, s in zip(self.norm_lbl[0], self.norm_lbl[1])],
                                                std=[1 / s for s in self.norm_lbl[1]])
                util.matplotlib_visualize(inv_normalize_img(images), inv_normalize_lbl(f_dis))
            elif self.norm_lbl is not None:
                inv_normalize_lbl = T.Normalize(mean=[-m / s for m, s in zip(self.norm_lbl[0], self.norm_lbl[1])],
                                                std=[1 / s for s in self.norm_lbl[1]])
                util.matplotlib_visualize(images, inv_normalize_lbl(f_dis))
            elif self.norm_img is not None:
                inv_normalize_img = T.Normalize(mean=[-m / s for m, s in zip(self.norm_img[0], self.norm_img[1])],
                                                std=[1 / s for s in self.norm_img[1]])
                util.matplotlib_visualize(inv_normalize_img(images), f_dis)
            else:
                util.matplotlib_visualize(images, f_dis)
            plt.show()

        return train_loader_list_, val_loader_list_

    def init_neptune(self) -> neptune.Run:
        if UNET_SEP:
            run = neptune.init(
                project="visuotactile-sensor/ir-unet-separate",
                api_token="TODO",
                source_files=['train.py']
            )
        else:
            run = neptune.init(
                project="visuotactile-sensor/interpretable-representations",
                api_token="TODO",
                source_files=['train.py']
            )

        config = dict(criterion=self.loss_fn, optimizer=self.optimizer,
                      scheduler=type(self.scheduler).__name__, augment=self.augment)
        net_config = dict(model=self.net)
        params = dict(folds=self.k, epochs=self.epochs, batch_size=self.batch_size, init_learning_rate=self.lr,
                      norm_img=self.norm_img, norm_lbl=self.norm_lbl)

        if isinstance(self.net, UNet):
            unet_config = dict(enc_chs=self.net.enc_chs, dec_chs=self.net.dec_chs, out_chs=self.net.out_chs,
                               pad=self.net.pad, pad_mode=self.net.pad_mode, resize_output=self.net.resize_output,
                               dropout=self.net.dropout, se_r=self.net.r)
            net_config.update(unet_config)
        if isinstance(self.net, UNetSep):
            unet_sep_config = dict(force_neurons=self.net.force_neurons)
            net_config.update(unet_sep_config)
        if isinstance(self.loss_fn, VistacLoss):
            params.update({'c_ifl': self.loss_fn.c_ifl})
            params.update({'mse_scale': self.loss_fn.mse_scale})
            params.update({'mode': self.loss_fn.mode})
        if isinstance(self.loss_fn, VistacLossSep):
            params.update({'mse_scale': self.loss_fn.mse_scale})
            params.update({'iou_scale': self.loss_fn.iou_scale})
            params.update({'focal': self.loss_fn.focal_params})

        run['config'] = config
        run['config/model'] = net_config
        run['parameters'] = params

        print()
        return run


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the U-Net on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--folds', '-f', metavar='val', type=int, default=FOLDS,
                        help='Number of folds for k-fold cross validation')
    parser.add_argument('--epochs', '-e', metavar='val', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='val', type=int, default=TRAIN_BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='val', type=float, default=LEARNING_RATE,
                        help='Initial learning rate', dest='lr')
    parser.add_argument('--load', '-m', metavar='model', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--device', '-d', metavar='dev', type=str, default=DEVICE, help='Set device: cpu / cuda')
    parser.add_argument('--neptune', '-n', default=False, action='store_true', help='Logging with Neptune')
    parser.add_argument('--validation', '-v', metavar='val', dest='val', type=float, default=VAL_PERCENT * 100,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--out-channels', '-c', dest='out_chs', metavar='val', type=int, default=OUTPUT_CHANNELS,
                        help='Number of output channels')
    # parser.add_argument('--workers', '-w', metavar='val', type=int, default=NUM_WORKERS,
    #                    help='Number of workers for the data loader')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    util.create_folder(str(MODELS_DIR), verbose=False)

    # Override parameters with parsed values if specified.
    FOLDS = args.folds
    EPOCHS = args.epochs
    TRAIN_BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    VAL_PERCENT = args.val / 100.0
    DEVICE = args.device
    NEPTUNE = args.neptune
    OUTPUT_CHANNELS = args.out_chs
    # NUM_WORKERS = args.workers

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Create a U-Net.
    if 'cuda' in DEVICE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(DEVICE)

    if UNET_SEP:
        unet = UNetSep(ENC_CHS, DEC_CHS, OUT_CHS, PADDING, PADDING_MODE, RESIZE_OUTPUT, DROPOUT, SE_BLOCK_R,
                       force_neurons=FORCE_NEURONS)
    else:
        unet = UNet(ENC_CHS, DEC_CHS, OUT_CHS, PADDING, PADDING_MODE, RESIZE_OUTPUT, DROPOUT, SE_BLOCK_R)

    if args.load:
        unet.load_state_dict(torch.load(args.load, map_location=device)['state_dict'])
        logging.info(f'{util.PrintColors.OKBLUE}Model loaded from {args.load}{util.PrintColors.ENDC}\n')

    logging.info(f'''Network:
        Encoder channels: {unet.enc_chs}
        Decoder channels: {unet.dec_chs}
        Output channels:  {unet.out_chs}
        Padding:          {unet.pad}
        Padding mode:     {unet.pad_mode}
        Resize output to: {unet.resize_output if unet.resize_output != (0, 0) else 'no resizing'}
        Dropout rate:     {unet.dropout}
        SE Block r:       {unet.r}
        Force neurons:    {unet.force_neurons if isinstance(unet, UNetSep) else None} \n''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP.
    optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    milestones = [m for m in range(2, EPOCHS, 2)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.75)

    if OPTIMIZER == 'RMSprop':
        optimizer = optim.RMSprop(unet.parameters(), lr=LEARNING_RATE, weight_decay=1e-8, momentum=0.9)

    if SCHEDULER == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=0, verbose=True)

    # Create an UNetTrainer.
    unet_trainer = UNetTrainer(unet, device, optimizer, scheduler, CRITERION,
                               norm_img=NORM_IMG, norm_lbl=NORM_LBL, k=FOLDS, augment=AUGMENT, mmap_mode=MMAP_MODE)

    # Start training.
    try:
        unet_trainer.start()
    except (KeyboardInterrupt, Exception):
        name_ = 'INTERRUPTED_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pth'
        unet_trainer.net.eval()
        state = {
            'state_dict': unet_trainer.net.state_dict(),
            'optimizer': unet_trainer.optimizer.state_dict()
        }
        torch.save(state, Path(MODELS_DIR, name_))
        print()
        logging.info(f"{util.PrintColors.OKBLUE}Saved interrupt as '{name_}'{util.PrintColors.ENDC}\n")
        raise

    sys.exit(0)
