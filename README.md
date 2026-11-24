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

## Learning curves

`python main.py` saves validation accuracy and macro F1 by epoch to
`runs/charts/learning_curves.png`. The figure includes every model and training method
selected for that run. Use `python main.py --show-charts` to also open the figure
in a Matplotlib window when training finishes.
