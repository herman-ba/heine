from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataConfig:
    path: str
    look_back: int
    horizon: int
    batch_size: int
    num_workers: int


def _masked_array(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    arr = df[columns].to_numpy(dtype=np.float32)
    return arr


def _create_windows(features: np.ndarray, close: np.ndarray, look_back: int, horizon: int,
                     split_idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    X_train, y_train, X_val, y_val = [], [], [], []
    total = len(features)
    for start in range(total - look_back - horizon + 1):
        end = start + look_back
        target_idx = end + horizon - 1
        window = features[start:end]
        target = close[target_idx]
        if not np.isfinite(window).all():
            continue
        if target_idx < split_idx:
            X_train.append(window)
            y_train.append(target)
        else:
            X_val.append(window)
            y_val.append(target)
    return (np.array(X_train), np.array(y_train)), (np.array(X_val), np.array(y_val))


class ForexDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.X[idx], self.y[idx]


def load_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    df = pd.read_parquet(cfg.path)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    df['rsi'] = ta.rsi(df['close'], length=14)

    features = ['open', 'high', 'low', 'close', 'volume', 'rsi']
    feat_arr = _masked_array(df, features)
    close_arr = df['close'].to_numpy(dtype=np.float32)

    split_idx = int(len(df) * 0.8)
    (X_train, y_train), (X_val, y_val) = _create_windows(feat_arr, close_arr, cfg.look_back, cfg.horizon, split_idx)

    train_ds = ForexDataset(X_train, y_train)
    val_ds = ForexDataset(X_val, y_val)

    pin_memory = torch.cuda.is_available()
    num_workers = cfg.num_workers if cfg.num_workers != -1 else max(1, os.cpu_count() // 2)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader
