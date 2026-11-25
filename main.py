import torch
import torch.nn as nn
from datasets import load_dataset


from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, default_collate

from encoder import Encoder, ENCODER_MODELS
from load_dataset import create_data_loaders, create_episodes

from feed_forward import MulticlassFFN
from lstm import MulticlassLSTM

from train_supervised import train_supervised
from train_RL import rl_train
from test_model import test_model
from metrics import plot_learning_curves

import argparse

from gpu_cleanup import format_peak_vram, free_gpu_resources

DATASET_ID = "siddharthmb/article-bias-prediction-media-splits"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_CHOICE = 2  # 1 = BERT, 2 = MPNet.
EMBEDDINGS_MODE = 1  # 1 = mean pooling, 2 = concatenate chunks.
MAX_EMBEDDING_CHUNKS = 8

SUPERVISED_EPOCHS = 10
RL_EPOCHS = 10
RL_ENTROPY_COEFFICIENT = 0.01
SUPERVISED_PRINT_INTERVAL = 2  # Print progress every N epochs.
RL_PRINT_INTERVAL = 10
LEARNING_RATE = 5e-4
EPISODE_LENGTH = 10
BATCH_SIZE = 32
SEED = 0

# LSTM hyperparameters
LSTM_PROJECTION_DIM = 64
LSTM_HIDDEN_DIM = 64
LSTM_NUM_HIDDEN_LAYERS = 1
LSTM_DROPOUT = 0.15
LSTM_BIDIRECTIONAL = False  # Let predictions use later articles in the episode.

# Data augmentation settings
AUGMENTATION_PROBABILITY = 0.0  # Chance to add noise to each training article.
AUGMENTATION_SIGMA = 0.005
SHUFFLE_EPISODES_EACH_EPOCH = False  # Shuffle articles before regrouping them.


def collate_episode(batch):
    inputs, labels = default_collate(batch)

    return inputs.unsqueeze(0), labels.unsqueeze(0)


def shuffled_episode_loader(dataset, epoch):
    generator = torch.Generator().manual_seed(SEED + epoch)
    episodes = create_episodes(dataset, episode_length=EPISODE_LENGTH, generator=generator)
    return DataLoader(episodes, batch_size=BATCH_SIZE, generator=generator)


def augment_embeddings(inputs, *, generator):
    mask = torch.rand(
        (*inputs.shape[:-1], 1), device=inputs.device, generator=generator
    ) < AUGMENTATION_PROBABILITY
    noise = torch.randn(
        inputs.shape, dtype=inputs.dtype, device=inputs.device, generator=generator
    )
    if EMBEDDINGS_MODE == 2:
        chunks = inputs.unflatten(-1, (MAX_EMBEDDING_CHUNKS, -1))
        present = chunks.ne(0).any(dim=-1, keepdim=True)
        noise = (noise.reshape_as(chunks) * present).reshape_as(inputs)
    perturbed = nn.functional.normalize(inputs + AUGMENTATION_SIGMA * noise, dim=-1)
    return torch.where(mask, perturbed, inputs)


def run_experiment(
    model_type,
    training_type,
    train_loader,
    valid_loader,
    test_loader,
    history=None
):

    torch.manual_seed(SEED)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    # Create model for the selected embedding width.
    embedding_dim = train_loader.dataset.tensors[0].size(-1)
    models = {
        "ffn": partial(MulticlassFFN, embedding_dim=embedding_dim),
        "lstm": partial(
            MulticlassLSTM,
            embedding_dim=embedding_dim,
            hidden_dim=LSTM_HIDDEN_DIM,
            n_layers=LSTM_NUM_HIDDEN_LAYERS,
            dropout_rate=LSTM_DROPOUT,
            bidirectional=LSTM_BIDIRECTIONAL,
            projection_dim=LSTM_PROJECTION_DIM
        )
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

    epoch_loader = None
    if SHUFFLE_EPISODES_EACH_EPOCH:
        epoch_loader = partial(shuffled_episode_loader, train_loader.dataset)
    augment_batch = None
    if AUGMENTATION_PROBABILITY > 0 and AUGMENTATION_SIGMA > 0:
        noise_generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        augment_batch = partial(augment_embeddings, generator=noise_generator)

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
            epochs=SUPERVISED_EPOCHS,
            validate_freq=len(episode_train_loader),
            print_interval=SUPERVISED_PRINT_INTERVAL,
            history=history,
            epoch_loader=epoch_loader,
            augment_batch=augment_batch
        )

    elif training_type == "rl":

        rl_train(
            model=model,
            train_loader=episode_train_loader,
            optimizer=optimizer,
            epochs=RL_EPOCHS,
            device=DEVICE,
            print_interval=RL_PRINT_INTERVAL,
            entropy_coefficient=RL_ENTROPY_COEFFICIENT,
            valid_loader=episode_valid_loader,
            history=history,
            epoch_loader=epoch_loader,
            augment_batch=augment_batch
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
    print(format_peak_vram(DEVICE))

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

    parser.add_argument(
        "--show-charts",
        action="store_true",
        help="Open the validation charts in a Matplotlib window after training."
    )
    args = parser.parse_args()

    if not 0 <= AUGMENTATION_PROBABILITY <= 1:
        parser.error("AUGMENTATION_PROBABILITY must be between 0 and 1")
    if not AUGMENTATION_SIGMA >= 0:
        parser.error("AUGMENTATION_SIGMA must be nonnegative")
    if not RL_ENTROPY_COEFFICIENT >= 0:
        parser.error("RL_ENTROPY_COEFFICIENT must be nonnegative")

    if ENCODER_CHOICE not in ENCODER_MODELS:
        parser.error("ENCODER_CHOICE must be 1 (BERT) or 2 (MPNet)")
    if EMBEDDINGS_MODE not in (1, 2):
        parser.error("EMBEDDINGS_MODE must be 1 (mean pooling) or 2 (concatenation)")
    model_name, max_length = ENCODER_MODELS[ENCODER_CHOICE]
    max_chunks = MAX_EMBEDDING_CHUNKS if EMBEDDINGS_MODE == 2 else None
    if max_chunks is not None and max_chunks < 1:
        parser.error("MAX_EMBEDDING_CHUNKS must be positive")
    cache_suffix = f"--concat-{max_chunks}" if max_chunks is not None else ""
    cache_name = model_name.replace("/", "--") + cache_suffix + ".pt"
    cache_path = Path(__file__).resolve().parent / "embeddings" / cache_name
    if cache_path.is_file():
        splits = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_loader, valid_loader, test_loader = (
            DataLoader(TensorDataset(*tensors), batch_size=8, shuffle=(index == 0))
            for index, tensors in enumerate(splits)
        )
        print(f"Loaded encodings from {cache_path}")
    else:
        encoder = Encoder(model_name, max_length, max_chunks=max_chunks)
        train_loader, valid_loader, test_loader = create_data_loaders(encoder, batch_size=8)
        del encoder
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            [loader.dataset.tensors for loader in (train_loader, valid_loader, test_loader)],
            cache_path
        )
        print(f"Saved encodings to {cache_path}")


    # Run experiments for all combinations of model and training method
    # python main.py --model lstm --method rl
    # python main.py --model lstm --method supervised
    # python main.py --model lstm --method supervised rl

    results = []
    histories = {}

    for model_type in args.model:

        for training_type in args.method:

            print(
                f"\n========== "
                f"{model_type.upper()} - "
                f"{training_type.upper()} "
                f"==========\n"
            )

            history = []
            test_acc, macro_f1 = run_experiment(
                model_type,
                training_type,
                train_loader,
                valid_loader,
                test_loader,
                history=history
            )
            if history:
                histories.setdefault(training_type, {})[model_type.upper()] = history
                final_validation = history[-1]
                print(
                    f"Sampled validation at epoch {final_validation.epoch}: "
                    f"Accuracy {final_validation.validation_accuracy:.2%} | "
                    f"Macro F1 {final_validation.validation_macro_f1:.4f}"
                )

            results.append((
                model_type,
                training_type,
                test_acc,
                macro_f1
            ))

    print("\n== FINAL TEST RESULTS ==\n")

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

    if histories:
        plot_learning_curves(histories, "runs/charts", show=args.show_charts)


if __name__ == "__main__":
    main()
