#!/usr/bin/env python3
"""
An implementation of the U-Net with Squeeze-and-Excitation blocks.

References:
    - [https://amaarora.github.io/2020/09/13/unet.html]
    - "U-Net: Convolutional Networks for Biomedical Image Segmentation" [https://arxiv.org/abs/1505.04597]
    - Squeeze-and-Excitation Block [https://amaarora.github.io/2020/07/24/SeNet.html]

Classes:
    - SEBlock
    - Block
    - Encoder
    - Decoder
    - UNet
"""
__author__ = 'Paul-Otto Müller'
__date__ = '13.03.2023'

import torch
import torchvision
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Tuple


class SEBlock(nn.Module):
    """
    An implementation of the Squeeze-and-Excitation Block.
    """

    def __init__(self, ch: int, r: int = 16) -> None:
        """
        Initializer.

        :param ch: Number of channels.
        :param r: Reduction ratio.
        :return: None
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class Block(nn.Module):
    """
    One block of the UNet consisting of two 3x3 'Conv2d' layers, each followed by a ReLU activation function.
    """

    def __init__(self, in_ch: int, out_ch: int, padding: str = 'valid',
                 padding_mode: str = 'zeros', r: int = 16) -> None:
        """
        Initializer.

        :param in_ch: Input channels.
        :param out_ch: Output channels.
        :param padding: 'valid': no padding, 'same': pads the input so the output has the shape as the input.
        :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
        :param r: SE reduction ratio.
        :return: None
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

    def forward(self, x: Tensor) -> Tensor:
        return self.se(self.block(x))


class Encoder(nn.Module):
    """
    The Encoder. The left part and contracting path of the U-shaped structure of the U-Net. It connects the blocks of
    two layers respectively by applying a 'MaxPool2d' operation with kernel size 2x2 and stride 2.
    """

    def __init__(self, chs: Tuple = (3, 64, 128, 256, 512, 1024), padding: str = 'valid', padding_mode: str = 'zeros',
                 dropout: float = 0.1, r: int = 16) -> None:
        """
        Initializer.

        :param chs: Tuple of numbers of encoder channels for each encoder level.
        :param padding: 'valid': no padding, 'same': pads the input so the output has the shape as the input.
        :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
        :param dropout: Dropout rate.
        :param r: SE reduction ratio.
        :return: None
        """
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], padding, padding_mode, r)
                                         for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: Tensor) -> List[Tensor]:
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
            x = self.dropout(x)
        return ftrs


class Decoder(nn.Module):
    """
    The Decoder. The right part and expansive path of the U-shaped structure of the U-Net. Every step of this path
    consists of an upsampling of the feature by applying a 2x2 'ConvTranspose2d' operation that halves the number of
    feature channels, a concatenation with the correspondingly cropped feature map from the contracting path,
    and two 3x3 'Conv2d' convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border
    pixels in every convolution, except 'padding=same' is set.
    """

    def __init__(self, chs: Tuple = (1024, 512, 256, 128, 64), padding: str = 'valid', padding_mode: str = 'zeros',
                 dropout: float = 0.1, r: int = 16) -> None:
        """
        Initializer.

        :param chs: Tuple of numbers of decoder channels for each decoder level.
        :param padding: 'valid': no padding, 'same': pads the input so the output has the shape as the input.
        :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
        :param dropout: Dropout rate.
        :param r: SE reduction ratio.
        :return: None
        """
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], padding, padding_mode, r)
                                         for i in range(len(chs) - 1)])
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: Tensor, encoder_features: List[Tensor]) -> Tensor:
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dropout(x)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs: Tensor, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    """
    The U-Net. A U-shaped neural network architecture consisting of a contracting path, the encoder, and an expansive
    path, the decoder. In the process, the encoder extracts a meaningful feature map from an input image, where
    the number of channels doubles at each step and halves the spatial dimension. Whereas the decoder up-samples
    the feature maps to retrieve spatial information, where at every step, it doubles the spatial dimension and
    halves the number of channels.
    """

    def __init__(self, enc_chs: Tuple = (3, 32, 64, 128, 256), dec_chs: Tuple = (256, 128, 64, 32), out_chs: int = 1,
                 padding: str = 'same', padding_mode: str = 'zeros', resize_output: Tuple = (0, 0),
                 dropout: float = 0.1, r: int = 16) -> None:
        """
        Initializer.

        :param enc_chs: Tuple of numbers of encoder channels for each encoder level.
        :param dec_chs: Tuple of numbers of decoder channels for each decoder level.
        :param out_chs: Number of output channels.
        :param padding: 'valid': no padding, 'same': pads the input so the output has the shape as the input.
        :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
        :param resize_output: (0, 0) if the output is not to be resized.
        :param dropout: Dropout rate.
        :param r: SE reduction ratio.
        :return: None
        """
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
        self.decoder = Decoder(dec_chs, padding=padding, padding_mode=padding_mode, dropout=dropout, r=r)
        self.head = nn.Conv2d(dec_chs[-1], out_chs, 1)
        self.out_sz = resize_output

        self.apply(self._init_layers)
        self.float()

    def forward(self, x: Tensor) -> Tensor:
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.out_sz != (0, 0):
            out = F.interpolate(out, self.out_sz)
        return out

    def _init_layers(self, module: nn.Module) -> None:
        """
        Initializes each model layer after Kaiming/He.

        :param module: A PyTorch module.
        :return: None
        """
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
