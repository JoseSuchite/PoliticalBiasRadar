from pathlib import Path
from typing import NamedTuple

import torch
from sklearn.metrics import accuracy_score, f1_score


CLASSES = (0, 1, 2)


class EpochMetrics(NamedTuple):
    epoch: int
    accuracy: float
    macro_f1: float


@torch.no_grad()
def evaluate_classifier(model, loader, device, criterion=None):
    """Collect article predictions and optionally compute average loss."""
    was_training = model.training
    model.eval()
    targets = []
    predictions = []
    total_loss = 0.0
    try:
        for inputs, labels in loader:
            output = model(inputs.to(device))
            labels = labels.reshape(-1)
            targets.extend(labels.tolist())
            predictions.append(output.argmax(dim=-1).reshape(-1))
            if criterion is not None:
                loss = criterion(output.reshape(-1, output.size(-1)), labels.to(device))
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


def plot_learning_curves(histories, output_path, show=False):
    """Save validation accuracy and macro F1 for each experiment in one figure."""
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    fig, (accuracy_ax, f1_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7), layout="constrained"
    )
    fig.suptitle("Validation learning curves")
    for label, history in histories.items():
        epochs, accuracies, f1_scores = zip(*history)
        marker_spacing = max(1, len(history) // 10)
        accuracy_ax.plot(epochs, accuracies, label=label, marker="o",
                         markersize=3, markevery=marker_spacing)
        f1_ax.plot(epochs, f1_scores, marker="o", markersize=3, markevery=marker_spacing)

    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    accuracy_ax.legend()
    f1_ax.set_ylabel("Macro F1")
    # Keep short supervised runs visible alongside much longer RL runs.
    last_epochs = [history[-1].epoch for history in histories.values()]
    if max(last_epochs) >= 10 * min(last_epochs):
        f1_ax.set_xscale("log")
        f1_ax.set_xlabel("Epoch (log scale)")
    else:
        f1_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        f1_ax.set_xlabel("Epoch")
    for ax in (accuracy_ax, f1_ax):
        ax.grid(alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=160)
        if show:
            plt.show()
    finally:
        plt.close(fig)
