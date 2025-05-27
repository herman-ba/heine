# LSTM Trainer

This module trains a stacked LSTM model on the `EURUSDT_15m.parquet` dataset.

## Installation

Dependencies are pinned in `requirements.txt`.

```bash
pip install -r trainer/requirements.txt
```

## Training

```bash
python trainer/train.py --config trainer/config.yaml
```

Training metrics are saved to `checkpoints/training.log`, which is cleared at
startup. Early stopping waits for 1000 epochs before counting patience, but the
best model may be found in any epoch.

The `config.yaml` file contains a `start_epoch` key that controls the early
stopping warmup period.

Arguments specified on the command line override values in the YAML config.

## Resume or Fine-tune

```bash
python trainer/train.py --checkpoint checkpoints/best.pt --epochs 500
```

After training, the script reloads the best checkpoint and prints the
validation loss and directional hit rate.
