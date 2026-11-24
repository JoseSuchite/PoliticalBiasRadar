import torch

# trains the ffn_model 
def train_supervised(
    model, train_loader, valid_loader, criterion, optimizer, device, epochs,
    validate_freq=2, print_interval=1
):

    # Move model to GPU/CPU
    model.to(device)

    # Count batches between validation checks
    counter = 0

    # Place model in training mode
    model.train()

    # Train for some number of epochs while evaluating
    for epoch in range(1, epochs + 1):
        # Batch loop
        for inputs, labels in train_loader:
            counter += 1

            # Move training data batch to GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # Get output from the model
            output = model(inputs)

            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))

            # Backpropagate
            loss.backward()

            # Update weights
            optimizer.step()

            # Validation
            if counter % validate_freq == 0:

                val_loss_total = 0.0
                val_samples = 0

                # Place model in evaluation mode
                model.eval()

                # Turn off gradient calculations
                with torch.no_grad():

                    for inputs, labels in valid_loader:

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        output = model(inputs)

                        val_loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))

                        val_loss_total += val_loss.item() * labels.numel()
                        val_samples += labels.numel()

                # Place model back in training mode
                model.train()

                average_val_loss = val_loss_total / val_samples

                if epoch % print_interval == 0 or epoch == epochs:
                    print(
                        f"Epoch: {epoch}/{epochs} | "
                        f"Step: {counter} | "
                        f"Train Loss: {loss.item():.6f} | "
                        f"Val Loss: {average_val_loss:.6f}"
                    )

