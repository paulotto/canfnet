#!/usr/bin/env python3
"""
An interface for accessing the DIGIT and GelSight Mini sensors.
"""
__author__ = 'Paul-Otto Müller'
__date__ = '13.03.2023'

import enum

import cv2
import numpy as np
from pathlib import Path
from digit_interface import Digit
from typing import Optional, Union, Dict, Tuple

from ..utils.utils import PrintColors, load_yaml


class TactileDevice(enum.Enum):
    DIGIT = 1
    GELSIGHTMINI = 2


class VistacInterface:
    """
    A class for interfacing a visuotactile sensor.
    """

    def __init__(self, device_name: Union[TactileDevice, str],
                 device_path: Optional[str] = '/dev/video0',
                 undistort_image: Optional[Path, str] = None) -> None:
        """
        Initializer.

        :param device_name: The name of the visuotactile device.
        :param device_path: The path to the visuotactile device.
        :param undistort_image: The path to a YAML file containing the camera parameters if the raw visuotactile
        image is to be undistorted.
        :return: None
        """
        self._file_dir: Path = Path(__file__).parent.resolve()
        self.cam_params_dict: Optional[Dict] = load_yaml(undistort_image) if undistort_image is not None else None

        self.device_name: Union[TactileDevice, str] = device_name
        self.device_path: Optional[str] = device_path
        self._device: Optional[cv2.VideoCapture, Digit] = None

        self.image: Optional[np.ndarray] = None
        self.image_dim: Tuple[int, int] = (320, 240)

    @property
    def device(self) -> Optional[cv2.VideoCapture, Digit]:
        return self._device

    def connect(self) -> None:
        """
        Connects the visuotactile camera.

        :return: None
        """
        self._device: cv2.VideoCapture = cv2.VideoCapture(self.device_path)
        if self._device is None or not self._device.isOpened():
            print(f"{PrintColors.WARNING} Failed to open camera at {self.device_path}! {PrintColors.ENDC}")

    def get_image(self) -> np.ndarray:
        """
        Gets an image from the visuotactile device by obtaining a raw image from the visuotactile camera, resizing it
        with the values specified through 'self.image_dim' and undistorting it if a path to a YAML file containing the
        camera parameters was specified.

        :return: A visuotactile image.
        """
        ret, image_ = self._device.read()
        if ret:
            self.image = cv2.resize(image_, self.image_dim)
            if self.cam_params_dict is not None:
                self.image = cv2.undistort(image_, self.cam_params_dict['camera_matrix'],
                                           self.cam_params_dict['dist_coeff'], None,
                                           self.cam_params_dict['new_camera_matrix'])
        else:
            self.image = None
            print(f"{PrintColors.FAIL} Failed to capture image! {self.device_path}! {PrintColors.ENDC}")

        return self.image


class DIGIT(VistacInterface):
    def __init__(self, serial_nr: str = 'D20025',
                 undistort_image: Optional[Path, str] = None) -> None:
        """
        Initializer.

        :param serial_nr: Set the serial number of the DIGIT sensor.
        :param undistort_image: The path to a YAML file containing the camera parameters if the raw visuotactile
        image is to be undistorted.
        :return: None
        """
        super().__init__(TactileDevice.DIGIT, None, undistort_image)
        self.serial_nr: str = serial_nr
        self.image_dim = (240, 320)

    def connect(self) -> None:
        """
        Connects the DIGIT camera.

        :return: None
        """
        self._device: Digit = Digit(self.serial_nr)
        self._device.connect()
        self._device.set_resolution(Digit.STREAMS["QVGA"])
        self._device.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])

    def get_image(self) -> np.ndarray:
        """
        Gets an image from the DIGIT sensor.

        :return: A visuotactile image from the DIGIT sensor.
        """
        self.image = self._device.get_frame()

        return self.image


class GelSightMini(VistacInterface):
    def __init__(self, device_path: Optional[str] = '/dev/video0', undistort_image: bool = True) -> None:
        """
        Initializer.

        :param device_path: The path to the visuotactile device.
        :param undistort_image: True if the raw visuotactile image is to be undistorted.
        :return: None
        """
        if undistort_image:
            super().__init__(TactileDevice.GELSIGHTMINI, device_path, Path(self._file_dir, 'params',
                                                                           'cam_params_gelsightmini.yaml'))
        else:
            super().__init__(TactileDevice.GELSIGHTMINI, device_path, None)

        self.image_dim = (320, 240)

    def get_image(self) -> np.ndarray:
        """
        Gets an image from the GelSight Mini sensor.

        :return: A visuotactile image from the GelSight Mini sensor.
        """
        ret, self.image = self._device.read()
        if ret:
            self.image = self.preprocess(self.image)
        else:
            print(f"{PrintColors.FAIL} Failed to capture image! {self.device_path}! {PrintColors.ENDC}")

        return self.image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the raw visuotactile image in a similar way than
        https://github.com/gelsightinc/gsrobotics/blob/aaf0cb55d8643bfbd0513cc1f106c6d7cbcb5504/gelsight/gsdevice.py#L65.
        Additionally, the image is undistorted if set to be true while initializing.

        :param image: A raw visuotactile image.
        :return: The preprocessed image.
        """
        cut_width: int = self.image_dim[0] // 7
        cut_height: int = self.image_dim[1] // 7

        image_ = cv2.resize(image, self.image_dim)

        if self.cam_params_dict is not None:
            image_ = cv2.undistort(image_, self.cam_params_dict['camera_matrix'],
                                   self.cam_params_dict['dist_coeff'], None,
                                   self.cam_params_dict['new_camera_matrix'])

        image_ = image_[cut_width:self.image_dim[0] - cut_width, cut_height:self.image_dim[1] - cut_height]
        image_ = cv2.resize(image_, self.image_dim)

        return image_
