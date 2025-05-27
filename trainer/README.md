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

During training, the script logs epoch metrics to `checkpoints/training.log`.
The file is overwritten at each run. Early stopping waits until the
`start_epoch` configured in `config.yaml` before counting patience,
but the best checkpoint can still come from any earlier epoch.

## Resume or Fine-tune

```bash
python trainer/train.py --checkpoint checkpoints/best.pt --epochs 500
```
