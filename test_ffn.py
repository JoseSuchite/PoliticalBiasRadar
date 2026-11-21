import torch

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

def ffn_test(model, test_loader, criterion, device):

    # Turn off gradient calculations
    with torch.no_grad():

        # Variables for tracking losses
        test_loss_total = 0.0
        total_samples = 0

        true_list = []
        pred_list = []

        # Place model in evaluation mode
        model.eval()

        # Run test data through model
        for inputs, labels in test_loader:

            # Move test data batch to GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get predicted output
            output = model(inputs)

            # Calculate loss
            test_loss = criterion(output, labels)

            test_loss_total += test_loss.item() * labels.size(0)
            total_samples += labels.size(0)

            # Get predicted class
            pred = torch.argmax(output, dim=1)

            # Place true and predicted labels in lists
            true_list += list(labels.cpu().tolist())
            pred_list += list(pred.cpu().tolist())

    print(confusion_matrix(true_list, pred_list, labels=[0, 1, 2]))
    print()

    print(classification_report(
        true_list,
        pred_list,
        labels=[0, 1, 2],
        target_names=["Left", "Center", "Right"],
        zero_division=0
    ))
    print()

    test_acc = accuracy_score(true_list, pred_list)
    macro_f1 = f1_score(
        true_list,
        pred_list,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    average_test_loss = test_loss_total / total_samples
    print(f"Test loss: {average_test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    return test_acc, macro_f1
