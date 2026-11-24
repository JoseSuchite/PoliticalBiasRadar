from sklearn.metrics import classification_report, confusion_matrix

from metrics import CLASSES, classification_scores, evaluate_classifier


def test_model(model, test_loader, criterion, device):
    model.eval()
    true_list, pred_list, average_test_loss = evaluate_classifier(
        model, test_loader, device, criterion
    )
    print(confusion_matrix(true_list, pred_list, labels=CLASSES))
    print()

    print(classification_report(
        true_list,
        pred_list,
        labels=CLASSES,
        target_names=["Left", "Center", "Right"],
        zero_division=0
    ))
    print()

    test_acc, macro_f1 = classification_scores(true_list, pred_list)
    print(f"Test loss: {average_test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2%}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    return test_acc, macro_f1
