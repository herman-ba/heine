from __future__ import annotations

import argparse
import os
import sys
import torch
from tqdm import tqdm

from callbacks import EarlyStopping, ModelCheckpoint, make_scheduler
from data.dataset import DataConfig, load_dataloaders
from models.lstm import LSTMModel, masked_mse
from utils.misc import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='trainer/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--symbol')
    parser.add_argument('--candle_interval')
    parser.add_argument('--look_back', type=int)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--device')
    parser.add_argument('--num_workers')
    parser.add_argument('--checkpoint_dir')
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg: dict[str, object] = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            if val.lower() in {'true', 'false'}:
                cfg[key] = val.lower() == 'true'
                continue
            try:
                cfg[key] = int(val.replace('_', ''))
                continue
            except ValueError:
                pass
            try:
                cfg[key] = float(val)
                continue
            except ValueError:
                pass
            cfg[key] = val.strip('"').strip("'")
    return cfg


def update_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    for k in cfg.keys():
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    return cfg


def resolve_device(device: str) -> str:
    if device != 'auto':
        return device
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = update_cfg(cfg, args)
    cfg['device'] = resolve_device(cfg.get('device', 'auto'))

    seed_everything()

    num_workers = -1 if cfg.get('num_workers') == 'auto' else int(cfg['num_workers'])
    data_cfg = DataConfig(
        path=os.path.join('data', f"{cfg['symbol']}_{cfg['candle_interval']}.parquet"),
        look_back=int(cfg['look_back']),
        horizon=int(cfg['horizon']),
        batch_size=int(cfg['batch_size']),
        num_workers=num_workers,
    )
    train_loader, val_loader = load_dataloaders(data_cfg)

    input_size = train_loader.dataset.X.shape[2]
    model = LSTMModel(input_size, int(cfg['hidden_size']), int(cfg['num_layers']), float(cfg['dropout']))
    model.to(cfg['device'])

    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']))
    scheduler = make_scheduler(optim)

    checkpoint = ModelCheckpoint(cfg['checkpoint_dir'])
    early_stop = EarlyStopping(int(cfg['patience']))

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=cfg['device']))

    best_epoch = 0
    for epoch in range(1, int(cfg['epochs']) + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(cfg['device']), y.to(cfg['device'])
            pred = model(x)
            loss = masked_mse(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(cfg['device']), y.to(cfg['device'])
                pred = model(x)
                loss = masked_mse(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        checkpoint.step(val_loss, model)
        if early_stop.step(val_loss):
            best_epoch = epoch
            break
        best_epoch = epoch

    best_path = os.path.join(cfg['checkpoint_dir'], 'best.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=cfg['device']))
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(cfg['device']), y.to(cfg['device'])
                pred = model(x)
                loss = masked_mse(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Best validation loss: {val_loss:.6f} at epoch {early_stop.best if early_stop.best < float('inf') else best_epoch}")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
