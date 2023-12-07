#!/usr/bin/env python3
"""
Utility functions.

Classes:
    - PrintColors

Functions:
    - find(s, ch) -> List[int]
    - visualize(**images) -> None
    - plot_img_and_mask(img, mask) -> None
"""
__author__ = 'Paul-Otto Müller'
__date__ = '12.09.2022'

import os
import yaml
import csv
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tikzplotlib
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import number
from typing import Optional, Union, List, Dict, Any, Tuple

numeric = Union[int, float, number]


class PrintColors:
    """
    Class for printing in different colors.

    Usage: print(f"{PrintColors.OKGREEN} <TEXT> {PrintColors.ENDC}")
    """
    HEADER = '\033[95m'
    PURPLE = '\033[95m'
    DARKCYAN = '\033[36m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def find(s, ch) -> List[int]:
    """
    Finds a character inside a string, and returns the corresponding indexes.

    :param s: A string.
    :param ch: A character to be searched for.
    :return: A list of corresponding indexes.
    """
    return [i for i, ltr in enumerate(s) if ltr == ch]


def load_yaml(file: str) -> Optional[Dict[str, Any]]:
    """
    Loads an YAML file.

    :param file: File name, including the path.
    :return: The loaded dictionary.
    """
    try:
        with open(file, 'r') as stream:
            loaded_dict = yaml.load(stream, Loader=yaml.FullLoader)
    except (yaml.YAMLError, IOError) as exc:
        print(f"Could not load yaml file {file}: {exc}")
        return None

    return loaded_dict


def read_csv(file: str) -> List[Dict[str, Any]]:
    """
    Reads a CSV file, and stores the rows inside a list of dictionaries,
    where the keys are defined by the first row inside the CSV file.

    :param file: File name including the path.
    """
    l_: List[Dict[str, Any]] = []
    file = file if '.csv' in file else file + '.csv'

    with open(file, mode='r', encoding='UTF8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            l_.append(dict(row))

    return l_


def write_csv(field_names: List[str], data: List[Dict[str, Any]], file: str, mode: str = 'w') -> None:
    """
    Writes a list of dictionaries to a CSV file, where the 'field_names'
    correspond to the keys of the dictionaries.

    :param field_names: List of the names of the fields.
    :param data: List of dictionaries containing the rows to be written.
    :param file: File name including the path.
    :param mode: Writing mode, i.e. 'w' - write, 'a' - append.
    """
    file = file if '.csv' in file else file + '.csv'
    mode = mode if mode == 'w' or mode == 'a' else 'a'
    write_header: bool = not Path(file).is_file()

    with open(file, mode, encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, lineterminator='\n', fieldnames=field_names)
        if write_header:
            writer.writeheader()
        writer.writerows(data)


def load_image(img_path: str, shape: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Loads an image.

    :param img_path: Path to the image to be loaded.
    :param shape: Shape to be used for resizing the image. No resizing if shape is None.
    :return: The loaded image.
    """
    img: np.ndarray = cv2.imread(img_path)     # flags=cv2.IMREAD_UNCHANGED
    if shape is not None:
        img = cv2.resize(img, shape)

    return img


def save_image(img_path: str, img: np.ndarray, img_format: str = '.jpg') -> None:
    """
    Writes an image to the specified path.

    :param img_path: The path where the image is to be stored.
    :param img: The image to be saved.
    :param img_format: The image format. '.jpg' or '.png'.
    """
    img_format = '.jpg' if img_format != '.jpg' or img_format != '.png' else img_format
    img_path += img_format if not img_path.endswith('.jpg') or not img_path.endswith('.png') else img_path
    cv2.imwrite(img_path, img)


def find_factors(nr_elements: int) -> List[int]:
    factors = []
    for i in range(1, nr_elements + 1):
        if nr_elements % i == 0:
            factors.append(i)
    return factors


def get_grid_size(nr_elements: int) -> Tuple[int, int]:
    factors = find_factors(nr_elements)
    nr_rows = factors[len(factors) // 2]
    nr_cols = nr_elements // nr_rows
    return nr_rows, nr_cols


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # result.shape = (height*nrows, width*ncols, intensity).
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def matplotlib_visualize(img: torch.Tensor,
                         f_dis: Optional[torch.Tensor] = None,
                         save_tex: Optional[str] = None) -> Tuple[matplotlib.figure.Figure]:
    """
    TODO
    """
    plt.style.use('ggplot')
    formatter_: str = "%1.3e N"
    cm_: cm = cm.get_cmap('RdYlBu_r', 20)   # coolwarm, BuPu, YlGnBu, viridis, GnBu, magma,
                                            # Set3, Set1, tab20b, Accent, tab10, Paired, Dark2, Pastel1
    fig = plt.figure(figsize=(15, 10))
    grid_subplots = plt.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    if img.dim() > 3:
        # img_grid = tv.utils.make_grid(img, nrow=img.size(0) // 3, normalize=False).permute(1, 2, 0).numpy()
        grid_size = get_grid_size(img.size(0))

        if f_dis is not None:
            # lbl_grid = gallery(f_dis.permute(0, 2, 3, 1).numpy(), img.size(0) // 3)
            lbl_grid = ImageGrid(fig, grid_subplots[0, 1],
                                 nrows_ncols=grid_size,
                                 axes_pad=(0.55, 0.15),
                                 label_mode="1",
                                 share_all=True,
                                 cbar_location="right",
                                 cbar_mode="each",
                                 cbar_size="7%",
                                 cbar_pad="2%")
            for axes, cax, f_dis_i in zip(lbl_grid, lbl_grid.cbar_axes,
                                          [f_dis[i] for i in range(f_dis.size(0))]):
                axes.grid(False)
                l_ = axes.imshow(f_dis_i.permute(1, 2, 0).numpy(), cmap=cm_)
                cb_ = cax.colorbar(l_, format=formatter_)
                # cb_.ax.set_ylabel('Force/pixel [N]')
                cb_.minorticks_on()

            if save_tex is not None:
                save_tex = save_tex if save_tex.endswith('.tex') else save_tex + '.tex'
                # tikzplotlib.clean_figure()
                tikzplotlib.save(save_tex)

            # fig.get_axes()[grid_size[1] // 2 + img.size(0) * 2].set_title('Normal Force Distribution [N/pixel]')
            fig.get_axes()[grid_size[1] // 2].set_title('Normal Force Distribution [N/pixel]')

        img_grid = ImageGrid(fig, grid_subplots[0, 0],
                             nrows_ncols=grid_size,
                             axes_pad=0.05,
                             label_mode="1")

        for axes, img_i in zip(img_grid, [img[i] for i in range(img.size(0))]):
            axes.grid(False)
            axes.imshow(img_i.permute(1, 2, 0).numpy())

        # fig.get_axes()[grid_size[1] // 2].set_title('Raw Tactile Image')
        fig.get_axes()[grid_size[1] // 2 + img.size(0) * 2].set_title('Raw Tactile Image')
    else:
        img_grid = img.permute(1, 2, 0).numpy()

        if f_dis is not None:
            lbl_grid = f_dis.permute(1, 2, 0).numpy()
            lbl_ax_ = fig.add_subplot(grid_subplots[0, 1])
            cb_ax_ = lbl_ax_.imshow(lbl_grid)
            cbar = fig.colorbar(cb_ax_, ax=lbl_ax_, format=formatter_,
                                fraction=0.044 * img_grid.shape[0] / img_grid.shape[1])
            cbar.minorticks_on()

            if save_tex is not None:
                save_tex = save_tex if save_tex.endswith('.tex') else save_tex + '.tex'
                tikzplotlib.clean_figure()
                tikzplotlib.save(save_tex)

            lbl_ax_.set_title('Normal Force Distribution [N/pixel]')

        img_ax_ = fig.add_subplot(grid_subplots[0, 0])
        img_ax_.set_title('Raw Tactile Image')
        img_ax_.grid(False)
        img_ax_.imshow(img_grid)

    return fig


def visualize_overlapped(img: torch.Tensor, label: Optional[torch.Tensor] = None) -> Tuple[plt.figure, plt.axes]:
    """
    TODO
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img.permute(1, 2, 0).numpy())

    if label is not None:
        ax.imshow(label.permute(1, 2, 0).numpy(), alpha=0.3)

    return fig, ax


def plot_img_and_mask(img, mask) -> None:
    """
    TODO
    """
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])


def create_folder(folder: str, verbose: bool = True) -> None:
    """
    Creates a folder if it doesn't exist yet.

    :param folder: Path of the folder including the folder name.
    :param verbose: Enable/Disable printing of status information.
    """
    if os.path.exists(folder):
        if verbose:
            print(f"{PrintColors.WARNING}[ContactAreaEstimation] Data folder {folder} already exists."
                  f"{PrintColors.ENDC}\n")
    else:
        try:
            os.mkdir(folder)
        except OSError:
            print(f"{PrintColors.FAIL}[ContactAreaEstimation] Failed to create the data folder {folder}!"
                  f"{PrintColors.ENDC}\n")
        else:
            if verbose:
                print(f"{PrintColors.OKGREEN}[ContactAreaEstimation] Successfully created the data folder "
                      f"{folder}.{PrintColors.ENDC}\n")
