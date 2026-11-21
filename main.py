
import torch

from bert_encoder import BERTEncoder
from load_dataset import create_data_loaders
from feed_forward import (ffn_model, ffn_criterion,ffn_optimizer, device)
from train_ffn import ffn_train
from test_ffn import ffn_test


def feed_foward_model(epochs=5):

    # Initialize BERT encoder
    encoder = BERTEncoder()

    # Create training, validation, and testing DataLoaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        encoder,
        batch_size=8
    )

    # Train FFN
    ffn_train(
        model=ffn_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=ffn_criterion,
        optimizer=ffn_optimizer,
        device=device,
        epochs=epochs,
        validate_freq=2
    )

    # Test FFN
    test_acc, macro_f1 = ffn_test(
        model=ffn_model,
        test_loader=test_loader,
        criterion=ffn_criterion,
        device=device
    )

    # Save model
    torch.save(ffn_model.state_dict(), "ffn_model.pth")

    return test_acc, macro_f1

test_acc, macro_f1 = feed_foward_model()

print(test_acc)
print(macro_f1)