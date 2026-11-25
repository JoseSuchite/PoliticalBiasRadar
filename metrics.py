from pathlib import Path
from typing import NamedTuple

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


CLASSES = (0, 1, 2)
EVALUATION_SEED = 0


class EpochMetrics(NamedTuple):
    epoch: int
    train_accuracy: float
    train_macro_f1: float
    validation_accuracy: float
    validation_macro_f1: float


@torch.no_grad()
def evaluate_classifier(model, loader, device, criterion=None):
    """Sample one prediction per article with a repeatable evaluation seed."""
    generator = torch.Generator(device=device).manual_seed(EVALUATION_SEED)
    was_training = model.training
    model.eval()
    targets = []
    predictions = []
    total_loss = 0.0
    try:
        for inputs, labels in loader:
            output = model(inputs.to(device))
            logits = output.reshape(-1, output.size(-1))
            labels = labels.reshape(-1)
            targets.extend(labels.tolist())
            probabilities = logits.softmax(dim=-1)
            predictions.append(torch.multinomial(probabilities, 1, generator=generator).squeeze(1))
            if criterion is not None:
                loss = criterion(logits, labels.to(device))
                total_loss += loss.item() * labels.numel()
    finally:
        model.train(was_training)

    if not targets:
        raise ValueError("Cannot evaluate an empty dataset.")
    average_loss = total_loss / len(targets) if criterion is not None else None
    return targets, torch.cat(predictions).cpu().tolist(), average_loss


def classification_scores(targets, predictions):
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, labels=CLASSES, average="macro", zero_division=0)
    return accuracy, macro_f1


def training_evaluation_loader(train_loader):
    """Score the training episodes without advancing the training sampler."""
    return DataLoader(
        train_loader.dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=train_loader.collate_fn,
        generator=torch.Generator().manual_seed(0),
    )


def epoch_metrics(epoch, model, train_eval_loader, device, val_targets, val_predictions):
    train_targets, train_predictions, _ = evaluate_classifier(model, train_eval_loader, device)
    return EpochMetrics(
        epoch,
        *classification_scores(train_targets, train_predictions),
        *classification_scores(val_targets, val_predictions),
    )


def plot_learning_curves(histories, output_dir, show=False):
    """Save separate supervised and RL figures for sampled train and validation scores."""
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    titles = {
        "supervised": "Supervised learning curves",
        "rl": "Reinforcement learning curves",
    }
    colors = {
        "train": {"FFN": "#8f3b0e", "LSTM": "#e28a3b"},
        "validation": {"FFN": "#174295", "LSTM": "#2b83bf"},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    try:
        for method, model_histories in histories.items():
            fig, (accuracy_ax, f1_ax) = plt.subplots(
                2, 1, sharex=True, figsize=(10, 7), layout="constrained"
            )
            figures.append(fig)
            fig.suptitle(f"{titles[method]}\nSampled predictions")
            for model, history in model_histories.items():
                epochs = [point.epoch for point in history]
                marker = "o" if len(history) <= 20 else None
                for ax, metric in ((accuracy_ax, "accuracy"), (f1_ax, "macro_f1")):
                    for split, style in (("train", "-"), ("validation", "--")):
                        values = [getattr(point, f"{split}_{metric}") for point in history]
                        ax.plot(
                            epochs, values, label=f"{model} {split}",
                            color=colors[split][model], linestyle=style, marker=marker, markersize=3
                        )
            accuracy_ax.set_ylabel("Accuracy")
            accuracy_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            accuracy_ax.legend(ncol=2)
            f1_ax.set_ylabel("Macro F1")
            f1_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            f1_ax.set_xlabel("Epoch")
            for ax in (accuracy_ax, f1_ax):
                ax.grid(alpha=0.25)

            output_path = output_dir / f"{method}_learning_curves.png"
            fig.savefig(output_path, dpi=160)
            print(f"Learning curves: {output_path.as_posix()}")
        if show:
            plt.show()
    finally:
        for fig in figures:
            plt.close(fig)
