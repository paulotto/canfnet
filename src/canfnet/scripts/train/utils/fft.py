#!/usr/bin/env python3
"""
Fast Fourier Transformation.

Classes:

Functions:
"""
__author__ = 'Paul-Otto Müller'
__date__ = '18.10.2022'

import numpy as np
import scipy.fft as fft
import matplotlib.figure
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from utils.util import read_csv


def plot_fft(data: np.ndarray, f_s: float) -> None:
    """
    Computes and displays the Discrete Fourier Transformation of the given data.

    :param data: The real-valued data to be transformed.
    :param f_s: The sampling frequency.
    """
    fourier = fft.rfft(data)
    freq = fft.rfftfreq(data.size, d=1./f_s)

    figure: matplotlib.figure.Figure = plt.figure()
    figure.canvas.manager.set_window_title('Discrete Fourier Transformation')
    grid_subplots = plt.GridSpec(2, 1, wspace=0.2, hspace=0.5)
    data_ax: plt.Axes = figure.add_subplot(grid_subplots[0, 0])
    fourier_ax: plt.Axes = figure.add_subplot(grid_subplots[1, 0])

    data_ax.plot(data)
    fourier_ax.plot(freq, abs(fourier))

    data_ax.set_title('Data')
    data_ax.set_xlabel('data points')
    data_ax.set_ylabel('f(x)')
    data_ax.grid(True)

    fourier_ax.set_title('DFT - Magnitude')
    fourier_ax.set_xlabel('frequency [Hz]')
    fourier_ax.set_ylabel('|DFT(.)|')

    plt.show()


def plot_force_fft(csv_file: str) -> None:
    force_l: List[Dict[str, Any]] = read_csv(csv_file)
    keys = list(force_l[0].keys())
    time = [float(row[keys[0]]) for row in force_l]
    force = np.array([float(row[keys[1]]) for row in force_l], dtype=object)

    dt_mean = [time[i] - time[i - 1] for i in range(1, len(time))]
    dt_mean = sum(dt_mean) / len(dt_mean)
    print(f"f_s: {1 / dt_mean}")

    plot_fft(force, 1 / dt_mean)
