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

Arguments specified on the command line override values in the YAML config.

Training progress is written to `checkpoints/training.log`, which is cleared at
startup. Early stopping begins counting patience only after `start_epoch`
(default: 1000) epochs.

## Resume or Fine-tune

```bash
python trainer/train.py --checkpoint checkpoints/best.pt --epochs 500
```
