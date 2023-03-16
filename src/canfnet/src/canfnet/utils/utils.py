#!/usr/bin/env python3
"""
Utility functions.

Classes:
    - PrintColors

Functions:
    - load_yaml(file: Optional[str, Path]) -> Optional[Dict[str, Any]]
    - write_yaml(data: Union[Mapping, Sequence, numeric], file: Optional[str, Path]) -> None
    - find(s, ch) -> List[int]
    - load_image(img_path: str, shape: Tuple[int, int, int] = None) -> NDArray[uint8]
    - save_image(img_path: str, img: NDArray[uint8], img_format: str = '.jpg') -> None
    - matplotlib_visualize(img: np.ndarray,
                         f_dis: Optional[np.ndarray] = None,
                         color_map: cm = cm.get_cmap('RdYlBu_r', 20),
                         norm: Optional[mpl.colors.Normalize] = mpl.colors.Normalize(vmin=-0.00075, vmax=-0.00012)
                         ) -> Tuple[matplotlib.figure.Figure]
    - preprocess(numpy_archive: str) -> List[np.ndarray]
"""
__author__ = 'Paul-Otto Müller'
__date__ = '13.03.2023'

import cv2
import yaml
import numpy as np
import matplotlib.figure
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Sequence, Mapping, Union, Any, Optional, List
from numpy import number

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


def load_yaml(file: Optional[Union[str, Path]]) -> Optional[Union[Dict[str, Any]]]:
    """
    Loads an YAML file.

    :param file: File name, including the path.
    :return: The loaded dictionary.
    """
    try:
        with open(file, 'r') as stream:
            loaded_dict = yaml.load(stream, Loader=yaml.FullLoader)
    except (yaml.YAMLError, IOError) as exc:
        print(f"{PrintColors.FAIL} Could not load yaml file {file}: {exc} {PrintColors.ENDC}")
        return None

    return loaded_dict


def write_yaml(data: Union[Mapping, Sequence, numeric], file: Optional[Union[str, Path]]) -> None:
    """
    Writes data to a YAML file.

    :param data: The data to be written.
    :param file: File name, including the path.
    :return: None
    """
    try:
        with open(file, 'w') as stream:
            yaml.dump(data, stream)
    except IOError as exc:
        print(f"{PrintColors.FAIL} Could not save data to yaml file {file}: {exc} {PrintColors.ENDC}")


def find(s, ch) -> List[int]:
    """
    Finds a character inside a string, and returns the corresponding indexes.

    :param s: A string.
    :param ch: A character to be searched for.
    :return: A list of corresponding indexes.
    """
    return [i for i, ltr in enumerate(s) if ltr == ch]


def load_image(img_path: str, shape: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Loads an image.

    :param img_path: Path to the image to be loaded.
    :param shape: Shape to be used for resizing the image. No resizing if shape is None.
    :return: The loaded image.
    """
    img: np.ndarray = cv2.imread(img_path)  # flags=cv2.IMREAD_UNCHANGED
    if shape is not None:
        img = cv2.resize(img, shape)

    return img


def save_image(img_path: str, img: np.ndarray, img_format: str = '.jpg') -> None:
    """
    Writes an image to the specified path.

    :param img_path: The path where the image is to be stored.
    :param img: The image to be saved.
    :param img_format: The image format. '.jpg' or '.png'.
    :return: None
    """
    img_format = '.jpg' if img_format != '.jpg' or img_format != '.png' else img_format
    img_path += img_format if not img_path.endswith('.jpg') or not img_path.endswith('.png') else img_path
    cv2.imwrite(img_path, img)


def matplotlib_visualize(img: np.ndarray,
                         f_dis: Optional[np.ndarray] = None,
                         color_map: cm = cm.get_cmap('RdYlBu_r', 20),
                         norm: Optional[mpl.colors.Normalize] = mpl.colors.Normalize(vmin=-0.00075, vmax=-0.00012)
                         ) -> Tuple[matplotlib.figure.Figure]:
    """
    Visualize an estimated force distribution together with its corresponding raw visuotactile image.

    :param img: The raw visuotactile image used as input for the estimation.
    :param f_dis: The corresponding normal force distribution.
    :param color_map: A matplotlib color map defining the display style.
    :param norm: Min and max values for linearly normalizing the normal force distribution into the [0.0, 1.0] interval.
    :return: The matplotlib figure objects.
    """
    plt.style.use('ggplot')
    formatter_: str = "%1.3e N"

    scalar_mappable_: cm.ScalarMappable = cm.ScalarMappable(norm=norm, cmap=color_map)
    fig_ = plt.figure(figsize=(15, 10))
    grid_subplots_ = plt.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    if f_dis is not None:
        lbl_ax_ = fig_.add_subplot(grid_subplots_[0, 1])
        f_dis_img_ = scalar_mappable_.to_rgba(f_dis, bytes=True, norm=True)
        cb_ax_ = lbl_ax_.imshow(f_dis_img_)
        cbar_ = fig_.colorbar(cb_ax_, ax=lbl_ax_, format=formatter_,
                              fraction=0.044 * img.shape[0] / img.shape[1])
        cbar_.minorticks_on()

        lbl_ax_.set_title('Normal Force Distribution [N/pixel]')

    img_ax_ = fig_.add_subplot(grid_subplots_[0, 0])
    img_ax_.set_title('Raw Tactile Image')
    img_ax_.grid(False)
    img_ax_.imshow(img)

    return fig_


def preprocess(numpy_archive: str) -> List[np.ndarray]:
    """
    Loads the given image numpy archive and returns the image data.
    The image data can thereby be compressed or uncompressed.

    :param numpy_archive: The path to a numpy archive.
    """
    image_list: List[np.ndarray] = []
    image_comp_dict: Dict[str, np.ndarray] = np.load(numpy_archive, mmap_mode='r')

    for _, img in image_comp_dict.items():
        image: np.ndarray = img

        if len(image.shape) != 3:
            image = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)

    return image_list
