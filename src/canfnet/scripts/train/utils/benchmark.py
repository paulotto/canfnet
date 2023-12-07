#!/usr/bin/env python3
import pytorch_benchmark.benchmark
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.benchmark
import pytorch_benchmark.benchmark
from pytorch_benchmark import benchmark
from tqdm import tqdm

from unet.unet import UNet


@torch.no_grad()
def test(model: nn.Module, device: torch.device = 'cuda', num_runs=300) -> None:
    model.eval()
    now = datetime.now()
    sample = torch.randn(10, 3, 320, 240)
    model.to(device=device)
    # sample = sample.to(device=device)

    result = benchmark(model, sample, num_runs=num_runs, print_fn=print)

    with open(f'./test_results/test_result_{now.strftime("%m-%d-%Y_%H-%M-%S")}', 'w') as file:
        yaml.dump(result, file)


def test_2(model: nn.Module, device: torch.device = 'cuda', num_runs=300) -> None:
    model.eval()
    now = datetime.now()
    sample = torch.randn(1, 3, 320, 240)
    model.to(device=device)
    sample = sample.to(device=device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_runs, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(sample)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(num_runs):
            starter.record()
            _ = model(sample)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / num_runs
    std_syn = np.std(timings)
    print(f"\n\n mean: {mean_syn}; std: {std_syn}")


if __name__ == '__main__':
    device: torch.device = torch.device('cuda')
    unet: UNet = UNet(padding='same', padding_mode='zeros', resize_output=(0, 0))
    unet.load_state_dict(torch.load('/home/paulmueller/training-interpretable-representations/models/'
                                    'model_04-11-2022_09-02-51_unet_mse_ifl_256.pth', map_location=device)['state_dict'])

    # test_2(unet, device=device, num_runs=300)
    test(unet, device=device, num_runs=300)
