from pathlib import Path
import pyarrow.parquet as pq
from datasets import load_dataset

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
        
        for label in sorted(indexedLabels()):
            balancedIndexes.extend(rng.sample(indexedLabels[label], minOccurance))
        rng.shuffle(balancedIndexes)

        data = {
            "content": [data["content"][i] for i in balanced_indices],
            "bias": [data["bias"][i] for i in balanced_indices],
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

def create_episodes(dataset, episode_length=10, shuffle=True):

    embeddings, labels = dataset.tensors

    if shuffle:
        indices = torch.randperm(len(labels))
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