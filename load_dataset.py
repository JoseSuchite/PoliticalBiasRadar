from pathlib import Path
import pyarrow.parquet as pq

import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "cleaned"


def load_articles(split="train"):
    filenames = {
        "train": "train-00000-of-00001.parquet",
        "valid": "valid-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet"
    }

    file_path = DATA_DIRECTORY / filenames[split]

    table = pq.read_table(
        file_path,
        columns=["content", "bias"]
    )

    return table.to_pydict()

def data_loader(data, encoder, batch_size=8, shuffle=True):
    embeddings_list = []
    labels_list = []

    # load data by batch so Bert doesnt use all the memory
    for start_of_batch in range(0, len(data["content"]), batch_size):
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