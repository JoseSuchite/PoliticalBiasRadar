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

## Embedding modes

Set `EMBEDDINGS_MODE` in `main.py` to choose how the encoder combines chunks.
Mode `1` averages the chunks into one vector per article and is the default.
Mode `2` concatenates up to `MAX_EMBEDDING_CHUNKS` chunk vectors in article
order. The default cap of eight produces 6,144 features per article. Shorter
articles are zero-padded, and longer ones keep their first chunks up to the
cap. Concatenated embeddings use a separate local cache.

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

## RL settings

Set `RL_ENTROPY_COEFFICIENT` in `main.py` to control the weight of the entropy
bonus during RL training. It defaults to `0.01`. Set it to `0.0` to disable
the bonus.

## Training augmentation

Set `AUGMENTATION_PROBABILITY` in `main.py` to the chance of adding Gaussian
noise to each training article embedding. `AUGMENTATION_SIGMA` is the noise
standard deviation. Set `SHUFFLE_EPISODES_EACH_EPOCH` to `True` to shuffle
articles before forming episodes each epoch. Both options are off by default.
Learning curves use clean embeddings and fixed evaluation episodes.

Noise is added per feature before the article vector is renormalized. A large
sigma can overwhelm the article signal, especially in concatenation mode.
Start with a small value such as `0.005` when comparing embedding modes.

## Learning curves

`python main.py` saves a separate figure for each selected training method under
`runs/charts/`: `supervised_learning_curves.png` and `rl_learning_curves.png`.
Both plot training and validation accuracy and macro F1 by epoch, using
seeded sampling from the same end-of-epoch weights for both splits. Final
test accuracy and macro F1 also use seeded sampling.
Use `python main.py --show-charts` to open the figures when training finishes.
