# CANFnet
<a href="#"><img src="https://img.shields.io/badge/python-v3.8+-blue.svg?logo=python&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.12.1-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="https://sites.google.com/view/canfnet"><img src="https://img.shields.io/badge/ROS-noetic-green.svg?logo=ros&style=for-the-badge" /></a>
<a href="https://sites.google.com/view/canfnet"><img src="https://img.shields.io/badge/Website-CANFnet-color?style=for-the-badge" /></a>

Visuotactile sensors are gaining momentum in robotics because they provide high-resolution contact measurements at 
a fraction of the price of conventional force/torque sensors. It is, however, not straightforward to extract useful 
signals from their raw camera stream, which captures the deformation of an elastic surface upon contact. To utilize 
visuotactile sensors more effectively, powerful approaches are required, capable of extracting meaningful 
contact-related representations. This work proposes a neural network architecture called CANFnet 
(Contact Area and Normal Force) that provides a high-resolution pixel-wise estimation of the contact area and normal 
force given the raw sensor images. The CANFnet is trained on a labeled experimental dataset collected using a 
conventional force/torque sensor, thereby circumventing material identification and complex modeling for label 
generation. We test CANFnet using commercially available DIGIT and GelSight Mini sensors and showcase its performance 
on real-time force control and marble rolling tasks. We are also able to report generalization of the CANFnets across 
different sensors of the same type. Thus, the trained CANFnet provides a plug-and-play solution for pixel-wise contact 
area and normal force estimation for visuotactile sensors. Additional information and videos can be seen at 
https://sites.google.com/view/canfnet.

## Table Of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Udev Rules](#udev-rules)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Prerequisites
- ROS Noetic: [installation instructions](http://wiki.ros.org/noetic/Installation)
- Needed Python packages can be installed with:
    ```bash
    pip3 install -r requirements.txt
    ```
- For Windows users, PyTorch (CUDA 11.7) can be installed as follows:
    ```bash
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
    ```

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/paulotto/canfnet.git /destination-directory/
    ```
2. Build the ROS workspace:
    ```bash
    cd /destination-directory/canfnet
    source /opt/ros/noetic/setup.bash
    catkin build
    ```
3. (Optional) ROS environment setup:

    Source the setup files automatically so that you don't have to source them every time a new shell is launched.
    ```bash
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    echo "source /destination-directory/canfnet/devel/setup.bash" >> ~/.bashrc
    ``` 

## Usage
Two nodes for publishing images from visuotactile sensors and for publishing the normal force and its 
distribution estimated by the CANFnet can be launched as follows:
```bash
roslaunch canfnet.launch rqt_gui:=true tactile_device:=GelSightMini tactile_device_path:=/dev/GelSightMini \
                         digit_serial:=D20025 tactile_cam_undistort:=true torch_device:=cuda canfnet_force_filt:=true \
                         model:="$(find canfnet)/models/model.pth"
```

| Argument              | Description                                                                                          |
|-----------------------|------------------------------------------------------------------------------------------------------|
| rqt_gui               | True if an rqt_gui window should be opened for displaying the data                                   |
| tactile_device        | The used visuotactile sensor (DIGIT or GelSightMini)                                                 |
| tactile_device_path   | The path to the visuotactile sensor (e.g. /dev/video0)                                               |
| digit_serial          | If a DIGIT sensor is used, the serial number of the sensor ('tactile_device_path' isn't needed then) |
| tactile_cam_undistort | True if the visuotactile image is to be undistorted                                                  |
| torch_device          | Either 'cpu' (CPU) or 'cuda' (GPU)                                                                   |
| canfnet_force_filt    | True if the estimated normal force is to be (median) filtered                                        |
| model                 | The PyTorch model including the file path                                                            |

## Models
Our trained models are placed inside [src/canfnet/models/](src/canfnet/models).

## Dataset
The data that has been used to train our model can be found 
[here](https://archimedes.ias.informatik.tu-darmstadt.de/s/6Jroz6Fqsr2faat).
The file [dataset.py](src/canfnet/src/canfnet/utils/dataset.py) contains a class *VistacDataSet* inheriting from 
PyTorch *Dataset* to access the visuotactile images and corresponding normal force distributions of the provided 
dataset. The class can be used as follows:

```python
from torch.utils.data import DataLoader
from canfnet.utils.dataset import VistacDataSet

dataset = VistacDataSet("/path/to/dataset/directory", norm_img=None, norm_lbl=None, augment=False, mmap_mode='r')
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for batch in dataloader:
    # visuotactile image, force distribution, object surface area [mm²], force
    img, f_dis, area, f = batch.values()
```

## Udev Rules
Optionally, you can give the visuotactile sensors more expressive names by writing the following in a 
*/etc/udev/rules.d/50-vistac.rules* file and adjusting the attributes:
```
SUBSYSTEM=="video4linux", SUBSYSTEMS=="usb", ATTR{name}=="DIGIT: DIGIT", ATTR{index}=="0", ATTRS{serial}=="D20025", SYMLINK+="DIGIT"
SUBSYSTEM=="video4linux", SUBSYSTEMS=="usb", ATTR{name}=="GelSight Mini R0B 28J3-CWXU: Ge", SYMLINK+="GelSightMini"
```
The corresponding device attributes can be found with the command:
```shell
udevadm info --name=/dev/video0 --attribute-walk
```
After adding the file, Udev can be reloaded with:
```shell
sudo udevadm control --reload
sudo udevadm trigger
```

## Project Structure
```
canfnet
│   README.md
│   requirements.txt   
└───src
    │   CMakeLists.txt
    └───canfnet
        │   CMakeLists.txt
        │   package.xml
        │   setup.py
        └───config
        │       canfnet.perspective        
        └───launch
        │       canfnet.launch
        └───models
        └───msg
        │       UNetEstimation.msg
        └───nodes
        │       canfnet_node.py
        │       visuotactile_sensor_node.py
        └───src
            └───canfnet
                │   __init__.py
                └───unet
                │       __init__.py
                │       predict.py
                │       unet.py
                └───utils
                │   │   dataset.py
                │   │   __init__.py
                │   │   utils.py
                │   └───params
                │           indenter_list_with_areas_in_mm.yaml
                └───visuotactile_sensor
                    │   __init__.py
                    │   visuotactile_interface.py
                    └───params
                        │   cam_params_digit.yaml
                        │   cam_params_gelsightmini.yaml
```

## Citation

If you use code or ideas from this work for your projects or research, please cite it.
```
@article{funk_canfnet,
title = {CANFnet: High-Resolution Pixelwise Contact Area and Normal Force Estimation for Visuotactile Sensors Using Neural Networks},
year = {2023},
url = {https://sites.google.com/view/canfnet},
author = {Niklas Funk and Paul-Otto Müller and Boris Belousov and Anton Savchenko and Rolf Findeisen and Jan Peters}
}
```