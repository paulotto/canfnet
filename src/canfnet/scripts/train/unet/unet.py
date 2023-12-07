#!/usr/bin/env python3
"""
U-Net.

Reference:
    - [https://amaarora.github.io/2020/09/13/unet.html]
    - "U-Net: Convolutional Networks for Biomedical Image Segmentation" [https://arxiv.org/abs/1505.04597]
    - Squeeze-and-Excitation Block [https://amaarora.github.io/2020/07/24/SeNet.html]

Classes:
    - Block
    - Encoder
    - Decoder
    - UNet
"""
__author__ = 'Paul-Otto Müller'
__date__ = '04.09.2022'

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from typing import Union, Optional, List, Tuple


class SEBlock(nn.Module):
    """
    TODO
    """

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class Block(nn.Module):
    """
    One block of the UNet consisting of two 3x3 'Conv2d' layers, each followed by a ReLU activation function.
    """

    def __init__(self, in_ch, out_ch, padding='valid', padding_mode='zeros', r=16) -> None:
        """
        Initializer.

        :param in_ch: Input channels.
        :param out_ch: Output channels.
        :param padding: 'valid': no padding, 'same': pads the input so the output has the shape as the input.
        :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
        """
        super().__init__()
        self.block = nn.Sequential(
            # Bias set to 'False' before 'BatchNorm', because the 'beta' of 'BatchNorm' has the same effect.
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=padding, padding_mode=padding_mode, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_ch, r=r)

    def forward(self, x):
        return self.se(self.block(x))


class DenseBlock(nn.Module):
    def __init__(self, in_ch, nr_blocks, padding='valid', padding_mode='zeros', r=16) -> None:
        super().__init__()
        self.nr_blocks = nr_blocks
        self.dense_blocks = nn.ModuleList([Block(i * in_ch, in_ch, padding, padding_mode, r)
                                           for i in range(1, nr_blocks + 1)])

    def forward(self, x):
        ftrs = []
        for block in self.dense_blocks:
            ftrs.append(x)
            if len(ftrs) > 1:
                for feature in ftrs[0:-1]:
                    x = torch.cat([x, feature], dim=1)
            x = block(x)
        return x


class Encoder(nn.Module):
    """
    The Encoder. The left part and contracting path of the U-shaped structure of the U-Net. It connects the blocks of
    two layers respectively by applying a 'MaxPool2d' operation with kernel size 2x2 and stride 2.
    """

    def __init__(self, chs=(3, 64, 128, 256, 512, 1024), padding='valid', padding_mode='zeros',
                 dropout=0.1, r=16) -> None:
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], padding, padding_mode, r)
                                         for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
            x = self.dropout(x)
        return ftrs


class EncoderMod(Encoder):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024), padding='valid', padding_mode='zeros',
                 dropout=0.1, r=16) -> None:
        super().__init__(chs, padding, padding_mode, dropout, r)
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], padding, padding_mode, r) for i in range(2)]
                                        + [Block(chs[i] + sum(chs[1:i]), chs[i + 1], padding, padding_mode, r)
                                           for i in range(2, len(chs) - 1)])

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            if len(ftrs) > 1:
                for feature in ftrs[0:-1]:
                    x = torch.cat([x, self.crop(feature, x)], dim=1)
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
            x = self.dropout(x)
        return ftrs

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class Decoder(nn.Module):
    """
    The Decoder. The right part and expansive path of the U-shaped structure of the U-Net. Every step of this path
    consists of an upsampling of the feature by applying a 2x2 'ConvTranspose2d' operation that halves the number of
    feature channels, a concatenation with the correspondingly cropped feature map from the contracting path,
    and two 3x3 'Conv2d' convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border
    pixels in every convolution, except 'padding=same' is set.
    """

    def __init__(self, chs=(1024, 512, 256, 128, 64), padding='valid', padding_mode='zeros', dropout=0.1, r=16) -> None:
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], padding, padding_mode, r)
                                         for i in range(len(chs) - 1)])
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dropout(x)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class DecoderMod(Decoder):
    def __init__(self, chs=(1024, 512, 256, 128, 64), padding='valid', padding_mode='zeros', dropout=0.1, r=16) -> None:
        super().__init__(chs, padding, padding_mode, dropout, r)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[0], chs[1], padding, padding_mode, r)]
                                        + [Block(chs[i] + sum(chs[0:i]) // 2, chs[i + 1], padding, padding_mode, r)
                                           for i in range(1, len(chs) - 1)])

    def forward(self, x, encoder_features):
        ftrs = []
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            ftrs.append(x)
            if i > 0:
                for feature in ftrs[0:-1]:
                    x = torch.cat([x, self.crop(feature, x)], dim=1)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dropout(x)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    """
    The U-Net. A U-shaped neural network architecture consisting of a contracting path, the encoder, and an expansive
    path, the decoder. In the process, the encoder extracts a meaningful feature map from an input image, where
    the number of channels doubles at each step and halves the spatial dimension. Whereas the decoder up-samples
    the feature maps to retrieve spatial information, where at every step, it doubles the spatial dimension and
    halves the number of channels.
    """

    def __init__(self, enc_chs=(3, 32, 64, 128, 256), dec_chs=(256, 128, 64, 32), out_chs=1,
                 padding='same', padding_mode='zeros', resize_output=(0, 0), dropout=0.1, r=16) -> None:
        super().__init__()
        self.enc_chs = enc_chs
        self.dec_chs = dec_chs
        self.out_chs = out_chs
        self.pad = padding
        self.pad_mode = padding_mode
        self.resize_output = resize_output
        self.dropout = dropout
        self.r = r

        self.encoder = Encoder(enc_chs, padding=padding, padding_mode=padding_mode, dropout=dropout, r=r)
        # self.dense_blocks = DenseBlock(enc_chs[-1], 3, padding, padding_mode, r)
        self.decoder = Decoder(dec_chs, padding=padding, padding_mode=padding_mode, dropout=dropout, r=r)
        self.head = nn.Conv2d(dec_chs[-1], out_chs, 1)
        self.out_sz = resize_output

        self.apply(self._init_layers)
        self.float()

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        # d_out = self.dense_blocks(enc_ftrs[::-1][0])
        # out = self.decoder(d_out, enc_ftrs[::-1][1:])
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.out_sz != (0, 0):
            out = F.interpolate(out, self.out_sz)
        return out

    def _init_layers(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')


class UNetSep(UNet):
    def __init__(self, enc_chs=(3, 32, 64, 128, 256), dec_chs=(256, 128, 64, 32), out_chs=1,
                 padding='same', padding_mode='zeros', resize_output=(0, 0), dropout=0.1, r=16,
                 force_neurons=(16, 8, 1)) -> None:
        super().__init__(enc_chs, dec_chs, out_chs, padding, padding_mode, resize_output, dropout, r)
        self.force_neurons = force_neurons
        self.dim_last_layer = (40, 30)  # H x W.
        self.max_force = torch.tensor([0.05])
        self.min_force = torch.tensor([-35.0])

        self.force_layers = nn.Sequential(
            nn.Linear(enc_chs[-1] * self.dim_last_layer[0] * self.dim_last_layer[1], self.force_neurons[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.force_neurons[0], self.force_neurons[1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.force_neurons[1], self.force_neurons[2])
        )

    def forward(self, x):
        enc_ftrs = self.encoder(x)

        # Force estimation layer.
        force_layer_input = enc_ftrs[::-1][0].view(enc_ftrs[::-1][0].size(0), -1)
        out_force = self.force_layers(force_layer_input)

        out_dec = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out_dec = self.head(out_dec)
        if self.out_sz != (0, 0):
            out_dec = F.interpolate(out_dec, self.out_sz)
        return out_force, out_dec
