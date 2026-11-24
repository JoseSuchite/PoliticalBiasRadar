# PoliticalBiasRadar

Our Dataset
https://huggingface.co/datasets/siddharthmb/article-bias-prediction-media-splits

Download the "train", "test", and "valid" parquet files.
https://huggingface.co/datasets/siddharthmb/article-bias-prediction-media-splits/tree/main/data

## Requirements
torch
transformers
numpy
pyarrow
pathlib
sklearn
matplotlib

## LSTM settings

Set `LSTM_PROJECTION_DIM` in `main.py` to control the width of the learned
projection applied to each article embedding before the LSTM. The default is
`64`. Set `LSTM_HIDDEN_DIM` to control the recurrent state width independently.
It also defaults to `64`. Set `LSTM_NUM_HIDDEN_LAYERS` to stack recurrent
layers. The default is `1`.

Set `LSTM_DROPOUT` to change the dropout rate applied to LSTM outputs before
classification during training. It defaults to `0.15`.

Set `LSTM_BIDIRECTIONAL` in `main.py` to `True` when the whole episode is
available before predictions. Each prediction can then use earlier and later
articles in that episode. The default is `False`.

## Training augmentation

Set `AUGMENTATION_PROBABILITY` in `main.py` to the chance of adding Gaussian
noise to each training article embedding. `AUGMENTATION_SIGMA` is the noise
standard deviation. Set `SHUFFLE_EPISODES_EACH_EPOCH` to `True` to shuffle
articles before forming episodes each epoch. Both options are off by default.
Learning curves use clean embeddings and fixed evaluation episodes.

## Learning curves

`python main.py` saves a separate figure for each selected training method under
`runs/charts/`: `supervised_learning_curves.png` and `rl_learning_curves.png`.
Both plot training and validation accuracy and macro F1 by epoch. The figures
use greedy predictions from the same end-of-epoch weights for both splits.
Use `python main.py --show-charts` to open the figures when training finishes.
