#!/usr/bin/env python3
"""
Predicting with a trained U-Net.
"""
__author__ = 'Paul-Otto Müller'
__date__ = '20.09.2022'

import argparse
import logging
from pathlib import Path
from datetime import datetime

import cv2
import onnx
import onnxruntime
import torch
import torch.onnx
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.figure
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from typing import Tuple, Optional, Union, List, Any
from torch import Tensor
from digit_interface import Digit

import utils.util as util
from unet.unet import UNet, UNetSep

# plt.style.use('ggplot')

PROJECT_DIR: Path = Path(__file__).parent.resolve()
MODELS_DIR: Path = Path(PROJECT_DIR, 'models')
ONNX_DIR: Path = Path(MODELS_DIR, 'onnx')
RESIZE: Tuple[int, int] = (0, 0)
DIGIT: str = 'D20025'
SAVE_TEX: Optional[str] = None
VISUALIZE: bool = True
LOG_FORCE: bool = False
UNET_SEP: bool = False
DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gelsight Mini.
NORM_IMG: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.4907543957, 0.4985137582, 0.4685586393]),
                                             Tensor([0.0307641067, 0.0246135872, 0.0398434214]))  # (mean, std).
NORM_LBL: Optional[Tuple[Tensor, Tensor]] = (Tensor([-6.0596092226e-05]), Tensor([0.0002053244])) # (mean, std).
# Digit.
'''
NORM_IMG: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.5024564266, 0.4860377908, 0.5020657778]),
                                             Tensor([0.0415902548, 0.0462468602, 0.0575232506]))  # (mean, std).
NORM_LBL: Optional[Tuple[Tensor, Tensor]] = (Tensor([-0.0001196197]), Tensor([0.0003911761]))     # (mean, std).
'''


def identity(x) -> Any:
    return x


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def init_digit(serial_nr: str = 'D20025') -> Digit:
    d = Digit(serial_nr)
    d.connect()
    d.set_resolution(Digit.STREAMS["QVGA"])
    d.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])
    return d


def normalize(image: np.ndarray, norm: Tuple[Tensor, Tensor], ordering: str = 'BGR') -> Tensor:
    if len(image.shape) < 2:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if 'BGR' in ordering:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    normalize_ = T.Compose([T.ToTensor(), T.Normalize(norm[0], norm[1])])
    return normalize_(image)


def unet_forward(unet_: UNet,
                 frames: Tensor,
                 device: torch.device = DEVICE) -> Tuple[Tensor, Tensor]:
    inv_normalize_dis = T.Normalize(mean=[-m / s for m, s in zip(NORM_DIS[0], NORM_DIS[1])],
                                    std=[1 / s for s in NORM_DIS[1]]) if NORM_DIS is not None else identity

    prediction_: Tensor = unet_.to(device)(frames)
    prediction_ = inv_normalize_dis(prediction_)
    f_dis_: Tensor = prediction_
    force_: Tensor = f_dis_.sum(dim=[2, 3])

    return force_.cpu(), f_dis_.cpu()


def unet_sep_forward(unet_sep: UNetSep,
                     frames: Tensor,
                     device: torch.device = DEVICE) -> Tuple[Tensor, Tensor]:
    force_, f_dis_ = unet_sep.to(device)(frames)
    f_dis_: Tensor = (torch.sigmoid(f_dis_) > 0.5)
    force_: Tensor = force_ * (unet_sep.max_force.to(device) - unet_sep.min_force.to(
        device)) + unet_sep.min_force.to(device)

    return force_.cpu(), f_dis_.cpu()


class ImageStream:
    """
    TODO
    """

    def __init__(self, digit: Digit,
                 net: Union[nn.Module, onnxruntime.InferenceSession],
                 dev: torch.device,
                 shape: Tuple[int, int] = (320, 240),
                 norm_img: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.3749360740, 0.3878611922, 0.3628277779]),
                                                              Tensor([0.1691615880, 0.0830215067, 0.0837985054])),
                 norm_dis: Optional[Tuple[Tensor, Tensor]] = (Tensor([-0.0001470775]), Tensor([0.0003999361])),
                 log_force: bool = False) -> None:
        self.digit: Digit = digit
        self.net: Union[nn.Module, onnxruntime.InferenceSession] = net
        self.device = dev
        self.shape = shape
        self.norm_img = norm_img
        self.norm_dis = norm_dis
        self.figure: matplotlib.figure.Figure = plt.figure(figsize=(10.3, 8))
        self.figure.canvas.manager.set_window_title('Visuotactile Sensor - Force Distribution')

        image_title: str = 'Tactile Image'
        # f_dis_title: str = 'Normal Force Distribution [$N/mm^{2}$]'
        f_dis_title: str = 'Normal Force Distribution [N/pixel]'
        f_title: str = 'Normal Force'
        grid_subplots = plt.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        self.img_ax: plt.Axes = self.figure.add_subplot(grid_subplots[0, 0])
        self.f_dis_ax: plt.Axes = self.figure.add_subplot(grid_subplots[0, 1])
        self.f_ax: plt.Axes = self.figure.add_subplot(grid_subplots[1, :])
        self.img_ax.set_title(image_title)
        self.f_dis_ax.set_title(f_dis_title)
        self.f_ax.set_title(f_title)
        self.f_ax.set_xlabel('data points')
        self.f_ax.set_ylabel('Force [N]')
        self.f_ax.grid(True)
        self.f_ax.text(0, 0, '')

        self.c_map: cm = cm.get_cmap('RdYlBu_r', 20)
        self.img: AxesImage = self.img_ax.imshow(np.zeros(shape + (3,)), cmap=self.c_map, animated=True)
        self.f_dis: AxesImage = self.f_dis_ax.imshow(np.zeros(shape + (1,)), cmap=self.c_map, animated=True)

        self.f_pts: int = 50
        self.f_x_axes: np.ndarray = np.arange(0, self.f_pts)
        self.f_list: List[np.ndarray] = [np.zeros(1)] * self.f_pts
        self.f, = self.f_ax.plot(self.f_x_axes, np.array(self.f_list, dtype=object), lw=1)

        self.cb: Colorbar = self.figure.colorbar(self.f_dis, ax=self.f_dis_ax, format="%1.3e N")
        self.cb.minorticks_on()

        self.anim = animation.FuncAnimation(self.figure, self.animate, frames=self.gen_function, init_func=self.init,
                                            interval=2, blit=True)
        """ Logging """
        # Saving to m4 using ffmpeg writer.
        # writer_video = animation.FFMpegWriter(fps=5)
        # self.anim.save(str(Path(PROJECT_DIR, 'vistac.mp4')), writer=writer_video)
        self.log_f: bool = log_force
        self.start: datetime = datetime.now()

        plt.show()

    def init(self) -> Tuple[AxesImage, AxesImage, plt.Axes]:
        self.img.set_data(np.zeros(self.shape + (3,)))
        self.f_dis.set_data(np.zeros(self.shape + (1,)))
        self.f.set_data([], [])

        return self.img, self.f_dis, self.f

    @torch.no_grad()
    def gen_function(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inv_normalize_dis = T.Normalize(mean=[-m / s for m, s in zip(self.norm_dis[0], self.norm_dis[1])],
                                        std=[1 / s for s in
                                             self.norm_dis[1]]) if self.norm_dis is not None else identity
        while True:
            frame_: np.ndarray = self.digit.get_frame()
            frame_norm_: Tensor = normalize(frame_, self.norm_img, 'BGR')
            frame_norm_ = frame_norm_.to(device=self.device).float()
            frame_norm_ = torch.unsqueeze(frame_norm_, dim=0)

            if isinstance(self.net, onnxruntime.InferenceSession):
                # TODO --> Needs testing!
                ort_inputs = {self.net.get_inputs()[0].name: to_numpy(frame_norm_)}
                ort_outs = self.net.run(None, ort_inputs)
                prediction_: np.ndarray = ort_outs[0][0]
                prediction_ = inv_normalize_dis(prediction_)
                f_dis_: np.ndarray = prediction_.transpose((1, 2, 0))
                force_: np.ndarray = np.sum(f_dis_)
            elif isinstance(self.net, UNetSep):
                self.net = self.net.to(device=self.device)
                force_, f_dis_ = self.net(frame_norm_)
                f_dis_: np.ndarray = (torch.sigmoid(f_dis_[0]) > 0.5).permute(1, 2, 0).cpu().numpy()
                force_: np.ndarray = (force_[0] * (self.net.max_force.to(self.device) - self.net.min_force.to(
                    self.device)) + self.net.min_force.to(self.device)).cpu().numpy()
            else:
                self.net = self.net.to(device=self.device)
                prediction_: Tensor = self.net(frame_norm_)
                prediction_ = inv_normalize_dis(prediction_)
                f_dis_: np.ndarray = prediction_[0].permute(1, 2, 0).cpu().numpy()
                force_: np.ndarray = np.sum(f_dis_)

            self.log_force(force_) if self.log_f else None

            yield frame_, f_dis_, force_

    def animate(self, array_tuple: Tuple[np.array, np.array, np.array]) -> Tuple[AxesImage, AxesImage, plt.Axes]:
        if not self.log_f:
            self.img.set_data(array_tuple[0])
            self.f_dis.set_data(array_tuple[1])
            self.f_dis.autoscale()
            self.f_dis.colorbar.update_normal(self.f_dis.colorbar.mappable)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        self.f_list.append(array_tuple[2])
        self.f_list.pop(0)
        self.f.set_data(self.f_x_axes, np.array(self.f_list, dtype=object))
        self.f_ax.texts.pop()
        self.f_ax.text(self.f_pts - 6, array_tuple[2] + 0.01, "F = {:06.4f} N".format(array_tuple[2]))
        y_min = np.min(np.array(self.f_list, dtype=object))
        y_max = np.max(np.array(self.f_list, dtype=object))
        self.f_ax.set_ylim(y_min, y_max, auto=True)

        return self.img, self.f_dis, self.f

    def log_force(self, force: np.ndarray, file_name: str = 'force') -> None:
        field_time: str = 'Time [s]'
        field_force: str = 'Force [N]'
        now_: datetime = datetime.now()
        t_: float = (now_ - self.start).total_seconds()
        force = {field_time: t_, field_force: force}

        file_: Path = Path(PROJECT_DIR, file_name + '_' + self.start.strftime("%d-%m-%Y_%H-%M-%S"))
        util.write_csv([field_time, field_force], [force], str(file_), 'a')


def torch_to_onnx(torch_model: nn.Module, out_file: str) -> onnxruntime.InferenceSession:
    torch_model.eval()
    test_input_ = torch.randn(1, 3, 320, 240, requires_grad=True)
    torch_out_ = torch_model(test_input_)
    torch.onnx.export(torch_model,  # Model being run.
                      test_input_,  # Model input (or a tuple for multiple inputs).
                      out_file,  # Where to save the model (can be a file or file-like object).
                      export_params=True,  # Store the trained parameter weights inside the model file.
                      # opset_version=13,          # The ONNX version to export the model to.
                      do_constant_folding=True,  # Whether to execute constant folding for optimization.
                      input_names=['image'],  # The model's input names.
                      output_names=['f_dis'],  # The model's output names.
                      dynamic_axes={'image': {0: 'batch_size', 2: 'H', 3: 'W'},  # Variable length axes.
                                    'f_dis': {0: 'batch_size', 2: 'H', 3: 'W'}})

    onnx_model_ = onnx.load(out_file)
    onnx.checker.check_model(onnx_model_)
    ort_session = onnxruntime.InferenceSession(out_file)

    # Compute ONNX Runtime output prediction.
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_input_)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare ONNX Runtime and PyTorch results.
    np.testing.assert_allclose(to_numpy(torch_out_), ort_outs[0], rtol=1e-03, atol=1e-05)
    logging.info(f"Exported model has been tested with ONNXRuntime, and is valid!")

    return ort_session


@torch.no_grad()
def predict(net: Union[UNet, UNetSep, onnxruntime.InferenceSession],
            images: Union[List[np.ndarray], Digit],
            dev: torch.device,
            norm_img: Optional[Tuple[Tensor, Tensor]] = (Tensor([0.3749360740, 0.3878611922, 0.3628277779]),
                                                         Tensor([0.1691615880, 0.0830215067, 0.0837985054])),
            norm_dis: Optional[Tuple[Tensor, Tensor]] = (Tensor([-0.0001470775]), Tensor([0.0003999361])),
            ordering: str = 'BGR') -> Tuple[np.ndarray, np.ndarray]:
    net.eval()
    predictions_: Tuple[Tensor, Tensor] = (torch.zeros(1), torch.zeros(1, 1, 1, 1))

    inv_normalize = T.Normalize(mean=[-m / s for m, s in zip(norm_img[0], norm_img[1])],
                                std=[1 / s for s in norm_img[1]]) if norm_img is not None else identity

    if isinstance(images, Digit):
        ImageStream(images, net, dev, norm_img=norm_img, norm_dis=norm_dis, log_force=LOG_FORCE)
    else:
        if NORM_IMG is not None:
            imgs_: List[Tensor] = [normalize(i, norm_img, ordering).to(device=dev).float() for i in images]
        else:
            if 'BGR' in ordering:
                images = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in images]
            imgs_: List[Tensor] = [torch.from_numpy(i.transpose((2, 0, 1))) for i in images]

        # TODO --> implementation for UNetSep.

        imgs_: Tensor = torch.stack(imgs_)

        if isinstance(net, onnxruntime.InferenceSession):
            # TODO --> Needs testing!
            pred_l_: List[Tensor] = []
            for i in imgs_:
                ort_inputs = {net.get_inputs()[0].name: to_numpy(i)}
                ort_outs = net.run(None, ort_inputs)
                prediction_np_: np.ndarray = ort_outs[0][0]
                pred_l_.append(Tensor(prediction_np_))
            predictions_: Tensor = torch.stack(pred_l_)
        elif isinstance(net, UNetSep):
            predictions_ = unet_sep_forward(net, imgs_, DEVICE)
        elif isinstance(net, UNet):
            predictions_ = unet_forward(net, imgs_, DEVICE)

        if VISUALIZE:
            util.matplotlib_visualize(inv_normalize(imgs_), predictions_[1], SAVE_TEX)
            plt.show()

    return predictions_[0].numpy(), predictions_[1].permute(0, 2, 3, 1).numpy()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict force distributions from input images')
    parser.add_argument('--model', '-m', default=MODELS_DIR, metavar='FILE', nargs='+',
                        help='Specify the file in which the model is stored', required=True)
    parser.add_argument('--onnx', '-O', action='store_true', default=False, help='Convert model to ONNX format')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='*', help='Filenames of input images')
    parser.add_argument('--output-tex', '-tex', metavar='OUTPUT', dest='tex', default=None,
                        help='Filename of the output tex file')
    parser.add_argument('--visualize', '-v', action='store_true', default=False,
                        help='Visualize the images as they are processed')
    parser.add_argument('--resize_output', '-r', dest='resize', nargs='*', default=RESIZE, metavar='H W',
                        help='Resize the output to HxW')
    parser.add_argument('--digit', '-d', action='store_true', default=False, help='Connect a Digit sensor')
    parser.add_argument('--serial', '-s', default=DIGIT, metavar='SERIAL',
                        help='Specify the serial number of the Digit sensor')
    parser.add_argument('--log', '-l', action='store_true', default=False, help='Logging the reconstructed force')
    parser.add_argument('--separate', '-sep', action='store_true', default=False,
                        help='Use the U-Net with two separate outputs.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    RESIZE = (int(args.resize[0]), int(args.resize[1]))
    VISUALIZE = args.visualize
    DIGIT = args.serial
    LOG_FORCE = args.log
    UNET_SEP = args.separate
    SAVE_TEX = args.tex
    VISUALIZE = True if SAVE_TEX is not None else VISUALIZE

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if '.onnx' in args.model[0]:
        unet: onnxruntime.InferenceSession = onnxruntime.InferenceSession(args.model[0])
    else:
        if UNET_SEP:
            NORM_DIS = None
            unet: UNetSep = UNetSep(padding='same', padding_mode='zeros', resize_output=RESIZE)
        else:
            unet: UNet = UNet(padding='same', padding_mode='zeros', resize_output=RESIZE)

        unet.to(device=DEVICE)
        unet.load_state_dict(torch.load(args.model[0], map_location=DEVICE)['state_dict'])
        unet.eval()

        if args.onnx:
            file_name_ = Path(args.model[0]).stem
            output_path_ = Path(ONNX_DIR, file_name_, '.onnx')
            unet: onnxruntime.InferenceSession = torch_to_onnx(unet, str(output_path_))

    logging.info(f'{util.PrintColors.OKBLUE}Model loaded from {args.model[0]}{util.PrintColors.ENDC}')
    logging.info(f'Using device {DEVICE}\n')

    if args.digit is not False:
        digit_ = init_digit(DIGIT)
        predict(unet, digit_, DEVICE, norm_img=NORM_IMG, norm_dis=NORM_DIS)
    else:
        imgs: List[np.ndarray] = []
        for img in args.input:
            imgs.append(util.load_image(img))
        predict(unet, imgs, DEVICE, norm_img=NORM_IMG, norm_dis=NORM_DIS)
