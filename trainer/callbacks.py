from __future__ import annotations

import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ModelCheckpoint:
    def __init__(self, directory: str) -> None:
        self.best = float('inf')
        self.path = os.path.join(directory, 'best.pt')
        os.makedirs(directory, exist_ok=True)

    def step(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best:
            self.best = val_loss
            torch.save(model.state_dict(), self.path)


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float('inf')
        self.count = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best:
            self.best = val_loss
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience


def make_scheduler(optimizer: torch.optim.Optimizer) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
