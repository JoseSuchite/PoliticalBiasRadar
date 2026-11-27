from pathlib import Path
import pyarrow.parquet as pq
from datasets import load_dataset
import random

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader

DATASET_ID = "siddharthmb/article-bias-prediction-media-splits"
HF_DATASET = load_dataset(DATASET_ID)


def load_articles(split="train", seed=42):
    data = HF_DATASET[split].select_columns(["content", "bias"]).to_dict()

    if split in ("train", "valid"):
        indexedLabels = {}

        for idx, label in enumerate(data["bias"]):
            indexedLabels.setdefault(label, []).append(idx)

        minOccurance = min(len(idxs) for idxs in indexedLabels.values())

        rng = random.Random(seed)
        balancedIndexes = []

        for label in sorted(indexedLabels.keys()):
            balancedIndexes.extend(rng.sample(indexedLabels[label], minOccurance))
        rng.shuffle(balancedIndexes)

        data = {
            "content": [data["content"][i] for i in balancedIndexes],
            "bias": [data["bias"][i] for i in balancedIndexes],
        }

    return data

def data_loader(data, encoder, batch_size=8, shuffle=True):
    embeddings_list = []
    labels_list = []

    # load data by batch so Bert doesnt use all the memory
    for start_of_batch in tqdm(range(0, len(data["content"]), batch_size), desc="Loading data", unit="batch"):
        end_of_batch = start_of_batch + batch_size

        articles = data["content"][start_of_batch:end_of_batch]
        labels = data["bias"][start_of_batch:end_of_batch]

        embeddings = encoder.encode_batch(articles)

        embeddings_list.append(embeddings)
        labels_list.extend(labels)

    all_embeddings = torch.cat(embeddings_list, dim=0).float()
    all_labels = torch.tensor(labels_list, dtype=torch.long)

    dataset = TensorDataset(all_embeddings, all_labels)

    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

def create_data_loaders(encoder, batch_size=8):

    # Load datasets
    train_data = load_articles("train")
    valid_data = load_articles("valid")
    test_data = load_articles("test")

    # Create training DataLoader
    train_loader = data_loader(
        train_data,
        encoder,
        batch_size=batch_size,
        shuffle=True
    )

    # Create validation DataLoader
    valid_loader = data_loader(
        valid_data,
        encoder,
        batch_size=batch_size,
        shuffle=False
    )

    # Create testing DataLoader
    test_loader = data_loader(
        test_data,
        encoder,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader

def balanced_episode_indices(labels, episode_length, shuffle=True, generator=None):
    per_class, remainder = divmod(episode_length, 3)
    if per_class < 1 or remainder:
        raise ValueError("Balanced episodes require a positive episode length divisible by 3.")

    indices = torch.randperm(len(labels), generator=generator) if shuffle else torch.arange(len(labels))
    ordered_labels = labels[indices]
    class_indices = [indices[ordered_labels == label] for label in range(3)]
    n_episodes = min(len(pool) for pool in class_indices) // per_class
    groups = torch.cat([
        pool[:n_episodes * per_class].reshape(n_episodes, per_class) for pool in class_indices
    ], dim=1)
    if shuffle:
        order = torch.rand(groups.shape, generator=generator).argsort(dim=1)
        groups = groups.gather(1, order)
    return groups


def create_episodes(
    dataset, episode_length=10, shuffle=True, generator=None, *, balance_classes=False
):
    embeddings, labels = dataset.tensors
    if balance_classes:
        groups = balanced_episode_indices(labels, episode_length, shuffle=shuffle, generator=generator)
        if len(groups) == 0:
            raise ValueError("Each class needs enough training articles for at least one episode.")
        return TensorDataset(embeddings[groups], labels[groups])

    if shuffle:
        indices = torch.randperm(len(labels), generator=generator)
        embeddings = embeddings[indices]
        labels = labels[indices]

    n_episodes = len(labels) // episode_length
    n_samples = n_episodes * episode_length

    embeddings = embeddings[:n_samples].reshape(
        n_episodes, episode_length, -1
    )

    labels = labels[:n_samples].reshape(
        n_episodes, episode_length
    )

    return TensorDataset(embeddings, labels)
