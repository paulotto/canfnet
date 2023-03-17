#!/usr/bin/env python3
"""
A class for accessing visuotactile images and corresponding force labels stored inside numpy archives.
"""
__author__ = 'Paul-Otto Müller'
__date__ = '16.03.2023'

import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Union, Optional

from canfnet.utils.utils import PrintColors, load_yaml


class VistacDataSet(Dataset):
    """
    A class for accessing visuotactile images and corresponding force labels stored inside numpy archives.
    If this dataset is used together with a PyTorch Dataloader, a number of workers greater than 1 can possibly lead
    to errors (https://github.com/numpy/numpy/issues/18124)!

    Note that this class needs a specific data structure for it to work. So have the numpy archives to be stored
    inside directories with the names of the corresponding objects used for creating the data.
    """

    def __init__(self, data_dir: str,
                 norm_img: Optional[Tuple[Tensor, Tensor]] = None,
                 norm_lbl: Optional[Tuple[Tensor, Tensor]] = None,
                 augment: bool = False,
                 mmap_mode: Optional[str] = 'r') -> None:
        """
        Initializer.

        :param data_dir: The path to the directory where the numpy archives are stored.
        :param norm_img: The mean and std deviation for each channel to normalize the input image.
        :param norm_lbl: The mean and std deviation to normalize the normal force distribution.
        :param augment: True if the input images should be augmented.
        :param mmap_mode: If None, the data will be loaded to memory. If 'r', the data will be memory-mapped read-only.
        For more information see https://numpy.org/doc/stable/reference/generated/numpy.memmap.html#numpy.memmap .
        :return: None
        """
        self.indenter_areas = load_yaml(str(Path(Path(__file__).parent, 'params',
                                                 'indenter_list_with_areas_in_mm.yaml').resolve()))

        train_data_files: Dict[str, Union[Dict[float, List[str]], List[str]]] = self.find_data_files(data_dir)
        self.input_files: Dict[float, List[str]] = train_data_files.get('inputs')
        self.label_files: List[str] = train_data_files.get('labels')
        self.ft_files: List[str] = train_data_files.get('ft')

        self.norm_img = norm_img
        self.norm_lbl = norm_lbl
        self.augment: bool = augment
        self.mmap_mode: Optional[str] = mmap_mode
        self._archive_elem_prefix: str = 'arr_'

        self.inputs: Dict[float, List[Any]] = self.input_files.copy()  # key = indenter area.
        for indenter_area, files in self.input_files.items():
            if mmap_mode is None:
                self.inputs[indenter_area] = []
                for f in files:
                    with np.load(f) as data:
                        self.inputs[indenter_area].append([self.preprocess(img, False, augment=self.augment,
                                                                           norm=self.norm_img)
                                                           for img in data.values()])
            else:
                self.inputs[indenter_area] = [np.load(f, mmap_mode=mmap_mode) for f in files]
        self.inputs = {k: v for k, v in self.inputs.items() if v}

        # Memory-mapped (mmap_mode) files cannot be larger than 2GB on 32-bit systems.
        if mmap_mode is None:
            self.labels: List[List] = []
            self.ft: List[List] = []
            for f in self.label_files:
                with np.load(f) as data:
                    self.labels.append([self.preprocess(lbl, True, norm=self.norm_lbl) for lbl in data.values()])
            for f in self.ft_files:
                with np.load(f) as data:
                    self.ft.append([torch.as_tensor(ft[0][2]) for ft in data.values()])
        else:
            self.labels: List[Any] = [np.load(f, mmap_mode=mmap_mode) for f in self.label_files]
            self.ft: List[Any] = [np.load(f, mmap_mode=mmap_mode) for f in self.ft_files]

        self._l_lbl: List[int] = [len(archive) for archive in self.labels]
        self._l_in: List[List[int]] = [[len(archive) for archive in self.inputs.get(key)] for key in self.inputs.keys()]

    def __len__(self) -> int:
        """
        Returns the number of samples inside the dataset.

        :return: The dataset length.
        """
        return sum([sum(l_) for l_ in self._l_in])

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        """
        Returns a dictionary containing the visuotactile images, the normal force distributions, the surface area of the
        used object in mm² and the measured total normal force.

        :param idx: The index indicating the position of a sample inside the dataset.
        :return: {'image': FloatTensor, 'force_distribution': FloatTensor,
                  'area': FloatTensor, 'force': FloatTensor}
        """
        idx_, archive_idx_, inputs_key_, inputs_archive_idx_ = self.get_index(idx)

        if self.mmap_mode is not None:
            input_ = self.preprocess(self.inputs[inputs_key_][inputs_archive_idx_][self._archive_elem_prefix
                                                                                   + str(idx_)], False,
                                     augment=self.augment, norm=self.norm_img)
            label_ = self.preprocess(self.labels[archive_idx_][self._archive_elem_prefix + str(idx_)], True,
                                     norm=self.norm_lbl)
            force_ = torch.as_tensor(self.ft[archive_idx_][self._archive_elem_prefix + str(idx_)][0][2])
        else:
            input_ = self.inputs[inputs_key_][inputs_archive_idx_][idx_]
            label_ = self.labels[archive_idx_][idx_]
            force_ = self.ft[archive_idx_][idx_]

        inputs_key_ = torch.as_tensor(inputs_key_)

        return {'image': input_.float(), 'force_distribution': label_.float(),
                'area': inputs_key_.float(), 'force': force_.float()}

    def get_index(self, idx) -> Tuple[int, int, float, int]:
        """
        Calculates an index pair for both the visuotactile images and forces and force distributions corresponding to
        the structure of the loaded list of numpy archives. The first index refers to an element within an archive,
        and the second index refers to the corresponding archive in the list of archives.

        :param idx: The current index used to access training data.
        :return: The index of an element inside a numpy archive and the index of which
        archive to be selected from the archive list. The first two indexes are for the forces and distributions, the
        second index pair for the visuotactile images.
        """
        l_lbl_idx_: int = 0
        keys_: List[float] = list(self.inputs.keys())
        key_: float = keys_[0]

        if idx >= self.__len__() or idx < 0:
            raise IndexError(f"{PrintColors.FAIL} Index ({idx}) out of range! {PrintColors.ENDC}")

        # Indexes for force distribution and force dictionaries.
        for i in range(1, len(self._l_lbl)):
            if idx >= sum(self._l_lbl[0:i]):
                l_lbl_idx_ += 1

        # Indexes for the input image dictionary.
        l_idx_: int = l_lbl_idx_
        for i, k in enumerate(keys_[1:]):
            if l_lbl_idx_ >= sum([len(l_i) for l_i in self._l_in[0:i + 1]]):
                key_ = k
                l_idx_ = l_lbl_idx_ - sum([len(l_i) for l_i in self._l_in[0:i + 1]])

        return idx - sum(self._l_lbl[0:l_lbl_idx_]), l_lbl_idx_, key_, l_idx_

    def find_data_files(self, data_dir: str) -> Dict[str, Union[Dict[float, List[str]], List[str]]]:
        """
        Finds all compressed numpy files containing visuotactile images and corresponding labels inside the
        specified directory.

        :param data_dir: Path of directory containing the data files.
        :return: A dictionary containing visuotactile image, normal force distribution and normal force data file paths
        ({'inputs': Union[Dict[float, List[str]], 'labels': List[str], }).
        """
        files_: Dict[str, Union[Dict[float, List[str]], List[str]]] = {'inputs': {}, 'labels': [], 'ft': []}

        for r, dirs, files in os.walk(data_dir):
            for indenter_dir in dirs:
                if indenter_dir in self.indenter_areas.keys():
                    files_['inputs'].update({self.indenter_areas[indenter_dir]: []})

                    for r_, bag_dirs, fs_ in os.walk(os.path.join(r, indenter_dir)):
                        for bag_dir in bag_dirs:
                            for root, d__, fs__ in os.walk(os.path.join(r_, bag_dir)):
                                for file in fs__:
                                    if '.npz' in file and 'input' in file:
                                        files_['inputs'][self.indenter_areas.get(indenter_dir)].append(
                                            os.path.join(root, file))
                                    if '.npz' in file and 'label' in file:
                                        files_['labels'].append(os.path.join(root, file))
                                    if '.npz' in file and 'ft' in file:
                                        files_['ft'].append(os.path.join(root, file))

        return files_

    @staticmethod
    def preprocess(np_array: np.ndarray,
                   is_label: bool,
                   augment: bool = False,
                   norm: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        """
        Preprocesses the training data. Loaded compressed image data is decoded if necessary, the arrays are formatted
        correctly and in the end converted to torch.Tensor's. Normalization to input images or data labels can be
        applied and optionally data augmentation (only for input images).

        :param np_array: A numpy array, either an image or a force distribution.
        :param is_label: Indicate whether the data is a label or not.
        :param augment: True if data augmentation is to be enabled.
        :param norm: (mean, std). A tuple of tensors used for normalizing the input images or data labels.
        :return: A tensor. Either the preprocessed image or label.
        """
        # Converts a numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor
        # of shape (C x H x W) in the range [0.0, 1.0] if the numpy.ndarray has dtype = np.uint8.
        to_tensor_ = T.Compose([T.ToTensor()])

        if not is_label:
            if len(np_array.shape) != 3:
                np_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            image_ = to_tensor_(np_array)

            if augment:
                image_ = VistacDataSet.augment(image_)

            if norm is not None:
                normalize_ = T.Compose([T.Normalize(norm[0], norm[1])])
                image_ = normalize_(image_)

            return image_

        if np_array.ndim == 2:
            np_array = np_array[np.newaxis, ...]

        label_ = torch.from_numpy(np_array)
        if norm is not None:
            normalize_ = T.Compose([T.Normalize(norm[0], norm[1])])
            label_ = normalize_(label_)

        return label_

    @staticmethod
    def augment(image: Tensor, norm: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        """
        Applies GaussianBlur and ColorJitter to the input image (can be adjusted according to need).
        Optionally, the image can be normalized with the given mean and std values.

        :param image: An input image.
        :param norm: Standardization parameters (mean, std) for each channel.
        :return: The processed image.
        """
        input_transforms = T.Compose([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
                                      T.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.25, hue=0.025)])
        image = input_transforms(image)
        if norm is not None:
            normalize_ = T.Compose([T.Normalize(norm[0], norm[1])])
            image = normalize_(image)

        return image
