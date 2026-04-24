import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for Python, NumPy, and (optionally) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional for non-training utilities.
        return
