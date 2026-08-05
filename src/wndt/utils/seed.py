"""Random seed utilities."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs used in this project (python, numpy, torch, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
