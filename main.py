import torch
import torch.nn as nn
from datasets import load_dataset


from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, default_collate

from bert_encoder import BERTEncoder, MODEL_NAME
from load_dataset import create_data_loaders, create_episodes

from feed_forward import MulticlassFFN
from lstm import MulticlassLSTM

from train_supervised import train_supervised
from train_RL import rl_train
from test_model import test_model
from metrics import plot_learning_curves

import argparse

from gpu_cleanup import free_gpu_resources

DATASET_ID = "siddharthmb/article-bias-prediction-media-splits"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPERVISED_EPOCHS = 10
RL_EPOCHS = 10
SUPERVISED_PRINT_INTERVAL = 2  # Print progress every N epochs.
RL_PRINT_INTERVAL = 10
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
    test_loader,
    history=None
):

    torch.manual_seed(SEED)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

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
            epochs=SUPERVISED_EPOCHS,
            validate_freq=len(episode_train_loader),
            print_interval=SUPERVISED_PRINT_INTERVAL,
            history=history
        )

    elif training_type == "rl":

        rl_train(
            model=model,
            train_loader=episode_train_loader,
            optimizer=optimizer,
            epochs=RL_EPOCHS,
            device=DEVICE,
            print_interval=RL_PRINT_INTERVAL,
            valid_loader=episode_valid_loader,
            history=history
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
    if DEVICE.type == "cuda":
        peak_vram_mib = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        print(f"Peak VRAM: {peak_vram_mib:.2f} MiB")
    else:
        print("Peak VRAM: N/A (CPU)")

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

    cache_name = MODEL_NAME.replace("/", "--") + ".pt"
    cache_path = Path(__file__).resolve().parent / "embeddings" / cache_name
    if cache_path.is_file():
        splits = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_loader, valid_loader, test_loader = (
            DataLoader(TensorDataset(*tensors), batch_size=8, shuffle=(index == 0))
            for index, tensors in enumerate(splits)
        )
        print(f"Loaded encodings from {cache_path}")
    else:
        encoder = BERTEncoder()
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
                label = f"{model_type.upper()} / {'RL' if training_type == 'rl' else 'Supervised'}"
                histories[label] = history

            results.append((
                model_type,
                training_type,
                test_acc,
                macro_f1
            ))

    print("\n== FINAL RESULTS ==\n")

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
        chart_path = "runs/charts/learning_curves.png"
        print(f"Validation charts: {chart_path}")
        plot_learning_curves(histories, chart_path, show=args.show_charts)


if __name__ == "__main__":
    main()
