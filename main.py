import torch
import torch.nn as nn
from datasets import load_dataset


from functools import partial
import math
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, default_collate

from encoder import Encoder, ENCODER_MODELS
from load_dataset import balanced_episode_indices, create_data_loaders, create_episodes

from feed_forward import MulticlassFFN
from lstm import MulticlassLSTM

from train_supervised import train_supervised
from train_RL import rl_train
from test_model import test_model
from metrics import CLASSES, EVALUATION_SEED, classification_scores, plot_learning_curves

import argparse

from gpu_cleanup import format_peak_vram, free_gpu_resources

DATASET_ID = "siddharthmb/article-bias-prediction-media-splits"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding settings
ENCODER_CHOICE = 1  # 1 = BERT, 2 = MPNet.
EMBEDDINGS_MODE = 1  # 1 = mean pooling, 2 = concatenate chunks.
MAX_EMBEDDING_CHUNKS = 8

# General Settings
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EPISODE_LENGTH = 12
BATCH_SIZE = 32
SEED = 0

# Reinforcement Learning Settings
STREAK_REWARD_WEIGHT = 0.2  # Weight for the longest streak reward in the total reward calculation.
RL_EPOCHS = 2000
RL_ENTROPY_COEFFICIENT = 0.005
RL_PRINT_INTERVAL = 25 # Print progress every N episodes.

# Supervised Learning Settings
SUPERVISED_EPOCHS = 12
SUPERVISED_PRINT_INTERVAL = 2  # Print progress every N epochs.

# LSTM hyperparameters
LSTM_PROJECTION_DIM = 32 # 6144 feature width for concatenate chunks, 768 for mean pooling
LSTM_HIDDEN_DIM = 32
LSTM_NUM_HIDDEN_LAYERS = 1
LSTM_DROPOUT = 0.1
LSTM_BIDIRECTIONAL = False  # Let predictions use later articles in the episode.

# Data augmentation settings
AUGMENTATION_PROBABILITY = 0.75  # Chance to add noise to each training article.
AUGMENTATION_SIGMA = 0.005
SHUFFLE_EPISODES_EACH_EPOCH = True  # Shuffle articles before regrouping them.
BALANCE_TRAINING_EPISODES = True  # Balance both training and validation episodes.

# Cyclical training settings
CYCLICAL_TRAINING = True
CYCLE_COUNT = 100  # Full waves per run. Use None to set a fixed period instead.
CYCLE_PERIOD = 10  # Epochs per wave when CYCLE_COUNT is None.
AUGMENTATION_MIN_SIGMA = 0.05
AUGMENTATION_MAX_SIGMA = 0.5
LEARNING_RATE_MIN = 1e-6
LEARNING_RATE_MAX = 1e-4
LEARNING_RATE_PHASE = math.pi


def collate_episode(batch):
    inputs, labels = default_collate(batch)

    return inputs.unsqueeze(0), labels.unsqueeze(0)


def balanced_validation_order(labels):
    counts = torch.bincount(labels, minlength=len(CLASSES))
    if len(counts) != len(CLASSES) or counts.min() == 0 or counts.unique().numel() != 1:
        raise ValueError("Balanced validation requires equal counts for Left, Center, and Right.")

    generator = torch.Generator().manual_seed(SEED)
    balanced = balanced_episode_indices(labels, EPISODE_LENGTH, generator=generator).flatten()
    unused = torch.ones_like(labels, dtype=torch.bool)
    unused[balanced] = False
    tail = unused.nonzero(as_tuple=True)[0]
    return torch.cat((balanced, tail[torch.randperm(len(tail), generator=generator)]))


def shuffled_episode_loader(dataset, epoch):
    generator = torch.Generator().manual_seed(SEED + epoch)
    episodes = create_episodes(
        dataset, episode_length=EPISODE_LENGTH, generator=generator,
        balance_classes=BALANCE_TRAINING_EPISODES
    )
    return DataLoader(episodes, batch_size=BATCH_SIZE, generator=generator)


def augment_embeddings(inputs, *, generator, sigma):
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
    perturbed = nn.functional.normalize(inputs + sigma * noise, dim=-1)
    return torch.where(mask, perturbed, inputs)


def oscillate(epoch, low, high, period, phase=0.0):
    angle = 2 * math.pi * (epoch - 1) / period + phase
    return low + (high - low) * ((1 - math.cos(angle)) / 2)


def prepare_cyclical_epoch(epoch, *, period, optimizer, augment_batch):
    sigma = oscillate(epoch, AUGMENTATION_MIN_SIGMA, AUGMENTATION_MAX_SIGMA, period)
    learning_rate = oscillate(
        epoch, LEARNING_RATE_MIN, LEARNING_RATE_MAX, period, LEARNING_RATE_PHASE
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    epoch_augment = None
    if augment_batch is not None and sigma > 0:
        epoch_augment = partial(augment_batch, sigma=sigma)
    return epoch_augment, f"LR: {learning_rate:.2e} | Sigma: {sigma:.4f}"


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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Create episodic training data
    torch.manual_seed(SEED)

    # Create episodic training data
    episode_train_loader = DataLoader(
        create_episodes(
            train_loader.dataset,
            episode_length=EPISODE_LENGTH,
            balance_classes=BALANCE_TRAINING_EPISODES
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    epoch_loader = None
    if SHUFFLE_EPISODES_EACH_EPOCH:
        epoch_loader = partial(shuffled_episode_loader, train_loader.dataset)
    augment_batch = None
    max_sigma = AUGMENTATION_MAX_SIGMA if CYCLICAL_TRAINING else AUGMENTATION_SIGMA
    if AUGMENTATION_PROBABILITY > 0 and max_sigma > 0:
        noise_generator = torch.Generator(device=DEVICE).manual_seed(SEED)
        augment_batch = partial(
            augment_embeddings, generator=noise_generator, sigma=AUGMENTATION_SIGMA
        )

    epochs = SUPERVISED_EPOCHS if training_type == "supervised" else RL_EPOCHS
    prepare_epoch = None
    if CYCLICAL_TRAINING:
        period = CYCLE_PERIOD if CYCLE_COUNT is None else epochs / CYCLE_COUNT
        prepare_epoch = partial(
            prepare_cyclical_epoch, period=period, optimizer=optimizer, augment_batch=augment_batch
        )

    validation_order = None
    if BALANCE_TRAINING_EPISODES:
        validation_order = balanced_validation_order(valid_loader.dataset.tensors[1]).tolist()

    episode_valid_loader = DataLoader(
        valid_loader.dataset,
        batch_size=EPISODE_LENGTH,
        shuffle=False,
        sampler=validation_order,
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
            epochs=epochs,
            validate_freq=len(episode_train_loader),
            print_interval=SUPERVISED_PRINT_INTERVAL,
            history=history,
            epoch_loader=epoch_loader,
            augment_batch=augment_batch,
            prepare_epoch=prepare_epoch
        )

    elif training_type == "rl":

        rl_train(
            model=model,
            train_loader=episode_train_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=DEVICE,
            print_interval=RL_PRINT_INTERVAL,
            entropy_coefficient=RL_ENTROPY_COEFFICIENT,
            streak_reward_weight=STREAK_REWARD_WEIGHT,
            valid_loader=episode_valid_loader,
            history=history,
            epoch_loader=epoch_loader,
            augment_batch=augment_batch,
            prepare_epoch=prepare_epoch,
            augmentation_sigma=AUGMENTATION_SIGMA
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


@torch.no_grad()
def compare_validation_episodes(validation_split, checkpoint, state=None):
    embeddings, labels = validation_split
    balanced = balanced_validation_order(labels)

    if state is None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    projection_dim, embedding_dim = state["projection.weight"].shape
    if embedding_dim != embeddings.size(-1):
        raise ValueError("The checkpoint input width does not match the embedding cache.")
    model = MulticlassLSTM(
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        hidden_dim=state["lstm.weight_hh_l0"].size(1),
        n_layers=sum(key.startswith("lstm.weight_hh_l") and not key.endswith("_reverse") for key in state),
        bidirectional="lstm.weight_hh_l0_reverse" in state,
    )
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    original = torch.arange(len(labels))

    print(f"Validation episode comparison for {checkpoint} ({len(labels)} articles)")
    for name, order in (("Existing", original), ("Balanced", balanced)):
        logits = torch.cat([
            model(embeddings[group].unsqueeze(0).to(DEVICE)).squeeze(0)
            for group in order.split(EPISODE_LENGTH)
        ])
        # Sample in the original article order so each article gets the same random draw.
        probabilities = logits[order.argsort().to(DEVICE)].softmax(dim=-1)
        sampler = torch.Generator(device=DEVICE).manual_seed(EVALUATION_SEED)
        predictions = torch.cat([
            torch.multinomial(batch, 1, generator=sampler).squeeze(1)
            for batch in probabilities.split(EPISODE_LENGTH)
        ]).cpu()
        accuracy, macro_f1 = classification_scores(labels.tolist(), predictions.tolist())
        left, center, right = torch.bincount(predictions, minlength=len(CLASSES)).tolist()
        print(
            f"{name:<8} | Accuracy: {accuracy:.2%} | Macro F1: {macro_f1:.4f} | "
            f"Predicted Left: {left} | Center: {center} | Right: {right}"
        )


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
    parser.add_argument(
        "--compare-validation-episodes",
        type=Path,
        metavar="LSTM_CHECKPOINT",
        help="Compare existing and balanced validation episodes using a saved LSTM checkpoint."
    )
    args = parser.parse_args()

    if not 0 <= WEIGHT_DECAY < math.inf:
        parser.error("WEIGHT_DECAY must be finite and nonnegative")
    if not 0 <= AUGMENTATION_PROBABILITY <= 1:
        parser.error("AUGMENTATION_PROBABILITY must be between 0 and 1")
    if not AUGMENTATION_SIGMA >= 0:
        parser.error("AUGMENTATION_SIGMA must be nonnegative")
    if not RL_ENTROPY_COEFFICIENT >= 0:
        parser.error("RL_ENTROPY_COEFFICIENT must be nonnegative")
    if CYCLICAL_TRAINING:
        if CYCLE_COUNT is None:
            if not 0 < CYCLE_PERIOD < math.inf:
                parser.error("CYCLE_PERIOD must be positive")
        elif type(CYCLE_COUNT) is not int or CYCLE_COUNT < 1:
            parser.error("CYCLE_COUNT must be a positive integer or None")
        if not 0 <= AUGMENTATION_MIN_SIGMA <= AUGMENTATION_MAX_SIGMA < math.inf:
            parser.error("Cyclical sigma bounds must be nonnegative and ordered")
        if not 0 < LEARNING_RATE_MIN <= LEARNING_RATE_MAX < math.inf:
            parser.error("Cyclical learning rate bounds must be positive and ordered")

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
    if args.compare_validation_episodes is not None:
        if not args.compare_validation_episodes.is_file():
            parser.error(f"LSTM checkpoint not found: {args.compare_validation_episodes}")
        state = torch.load(args.compare_validation_episodes, map_location="cpu", weights_only=True)
        width = state["projection.weight"].size(1)
        stem = model_name.replace("/", "--")
        candidates = [
            cache_path,
            cache_path.with_name(stem + ".pt"),
            *sorted(cache_path.parent.glob(stem + "--concat-*.pt")),
        ]
        for path in dict.fromkeys(candidates):
            if path.is_file():
                _, validation_split, _ = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                if validation_split[0].size(-1) == width:
                    compare_validation_episodes(validation_split, args.compare_validation_episodes, state)
                    return
        parser.error(f"No {width}-dimensional embedding cache for {model_name}")

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


    # python main.py -> Run experiments for all combinations of model and training method
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
