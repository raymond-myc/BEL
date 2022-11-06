import logging
import math
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def setup_std_logger(log_dir: str, print_to_console: bool = True):
    std_logger = logging.getLogger()
    std_logger.setLevel(logging.INFO)

    std_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, "std_output.log"))

    if print_to_console:
        std_logger.addHandler(std_handler)
    std_logger.addHandler(file_handler)

    return std_logger


def setup_tb_logger(log_dir: str):
    tb_log_dir = os.path.join(log_dir, "tb_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_logger = SummaryWriter(log_dir=tb_log_dir)
    return tb_logger


def calc_acts_delta(acts_1: torch.Tensor, acts_2: torch.Tensor):
    delta = torch.mean(torch.linalg.norm(acts_1 - acts_2, dim=-1), dim=0) / (
        math.sqrt(acts_2.shape[-1])
    )
    return delta


def rank_tensor_elements(x: torch.Tensor, descending: bool = True):
    sorted_idx = x.argsort(descending=descending)
    ranks = torch.empty_like(sorted_idx)
    ranks[sorted_idx] = torch.arange(len(x))
    return ranks

def partitioned_sample_uni(n_samples: int, upper_bound: float):
    """
    Uniformly sample n samples from [0, upper_bound] interval in a stable way
    """
    samples = []
    for i in range(n_samples):
        delta_interval = upper_bound / n_samples
        uni_lbs = np.arange(0, upper_bound, delta_interval)
        uni_ubs = np.arange(
            delta_interval, upper_bound + delta_interval, delta_interval
        )
        high=uni_ubs[i]
        low=uni_lbs[i]
        x =(np.random.rand() - 0.5) * (high - low) + (high + low) / 2
        x = np.clip(x, a_min=1e-2, a_max=upper_bound)
        samples.append(x)
    random.shuffle(samples)
    return samples