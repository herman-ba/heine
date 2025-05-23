from __future__ import annotations

import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
