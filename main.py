import torch
import torch.nn as nn
from datasets import load_dataset


from torch.utils.data import DataLoader, default_collate

from bert_encoder import BERTEncoder
from load_dataset import create_data_loaders, create_episodes

from feed_forward import MulticlassFFN
from lstm import MulticlassLSTM

from train_supervised import train_supervised
from train_RL import rl_train
from test_model import test_model

import argparse

from gpu_cleanup import free_gpu_resources

DATASET_ID = "siddharthmb/article-bias-prediction-media-splits"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 5
LEARNING_RATE = 5e-4
EPISODE_LENGTH = 10
BATCH_SIZE = 32
SEED = 0


def collate_episode(batch):
    inputs, labels = default_collate(batch)

    return inputs.unsqueeze(0), labels.unsqueeze(0)


def run_experiment(
    model_type,
    training_type,
    train_loader,
    valid_loader,
    test_loader
):

    torch.manual_seed(SEED)

    # Create model
    models = {
        "ffn": MulticlassFFN,
        "lstm": MulticlassLSTM
    }

    model = models[model_type]().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Create episodic training data
    torch.manual_seed(SEED)

    # Create episodic training data
    episode_train_loader = DataLoader(
        create_episodes(
            train_loader.dataset,
            episode_length=EPISODE_LENGTH
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    # Preserve all validation and testing articles
    episode_valid_loader = DataLoader(
        valid_loader.dataset,
        batch_size=EPISODE_LENGTH,
        shuffle=False,
        collate_fn=collate_episode
    )

    episode_test_loader = DataLoader(
        test_loader.dataset,
        batch_size=EPISODE_LENGTH,
        shuffle=False,
        collate_fn=collate_episode
    )

    # Train model
    if training_type == "supervised":

        train_supervised(
            model=model,
            train_loader=episode_train_loader,
            valid_loader=episode_valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epochs=EPOCHS,
            validate_freq=len(episode_train_loader)
        )

    elif training_type == "rl":

        rl_train(
            model=model,
            train_loader=episode_train_loader,
            optimizer=optimizer,
            epochs=EPOCHS,
            device=DEVICE
        )

    else:
        raise ValueError("Invalid training type")

    # Test model
    test_acc, macro_f1 = test_model(
        model=model,
        test_loader=episode_test_loader,
        criterion=criterion,
        device=DEVICE
    )

    # Save model
    torch.save(
        model.state_dict(),
        f"{model_type}_{training_type}.pth"
    )

    return test_acc, macro_f1


def main():
    # Free GPU resources before training and testing
    free_gpu_resources()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["ffn", "lstm"],
        nargs="+",
        default=["ffn", "lstm"]
    )

    parser.add_argument(
        "--method",
        choices=["supervised", "rl"],
        nargs="+",
        default=["supervised", "rl"]
    )

    args = parser.parse_args()

    # Generate BERT embeddings once
    encoder = BERTEncoder()

    train_loader, valid_loader, test_loader = create_data_loaders(
        encoder,
        batch_size=8
    )

    del encoder


    # Run experiments for all combinations of model and training method
    # python main.py --model lstm --method rl
    # python main.py --model lstm --method supervised
    # python main.py --model lstm --method supervised rl

    results = []

    for model_type in args.model:

        for training_type in args.method:

            print(
                f"\n========== "
                f"{model_type.upper()} - "
                f"{training_type.upper()} "
                f"==========\n"
            )

            test_acc, macro_f1 = run_experiment(
                model_type,
                training_type,
                train_loader,
                valid_loader,
                test_loader
            )

            results.append((
                model_type,
                training_type,
                test_acc,
                macro_f1
            ))

    print("\n== RESULTS ==\n")

    print(
        f"{'Model':<12}"
        f"{'Method':<15}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
    )

    print("-" * 51)

    for model, method, acc, f1 in results:

        print(
            f"{model.upper():<12}"
            f"{method:<15}"
            f"{acc:>12.2%}"
            f"{f1:>12.4f}"
        )

    print("-" * 51)

    # Free GPU resources after training and testing
    free_gpu_resources()


if __name__ == "__main__":
    main()