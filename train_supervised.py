from gpu_cleanup import format_peak_vram
from metrics import classification_scores, epoch_metrics, evaluate_classifier, training_evaluation_loader

# trains the ffn_model 
def train_supervised(
    model, train_loader, valid_loader, criterion, optimizer, device, epochs,
    validate_freq=2, print_interval=1, history=None, epoch_loader=None, augment_batch=None,
    prepare_epoch=None
):

    # Move model to GPU/CPU
    model.to(device)

    # Count batches between validation checks
    counter = 0

    # Place model in training mode
    model.train()
    train_eval_loader = training_evaluation_loader(train_loader) if history is not None else None

    # Train for some number of epochs while evaluating
    for epoch in range(1, epochs + 1):
        epoch_augment, schedule = (
            prepare_epoch(epoch) if prepare_epoch is not None else (augment_batch, "")
        )
        schedule_suffix = f" | {schedule}" if schedule else ""
        # Batch loop
        batches = epoch_loader(epoch) if epoch_loader is not None else train_loader
        for batch, (inputs, labels) in enumerate(batches, start=1):
            counter += 1

            # Move training data batch to GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            if epoch_augment is not None:
                inputs = epoch_augment(inputs)

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

            validate = counter % validate_freq == 0
            record_epoch = train_eval_loader is not None and batch == len(batches)
            if validate or record_epoch:

                targets, predictions, average_val_loss = evaluate_classifier(
                    model, valid_loader, device, criterion if validate else None
                )
                if record_epoch:
                    history.append(epoch_metrics(
                        epoch, model, train_eval_loader, device, targets, predictions
                    ))
                if validate and (epoch % print_interval == 0 or epoch == epochs):
                    if record_epoch:
                        validation_accuracy = history[-1].validation_accuracy
                    else:
                        validation_accuracy, _ = classification_scores(targets, predictions)
                    print(
                        f"Epoch: {epoch}/{epochs} | "
                        f"Step: {counter} | "
                        f"Train Loss: {loss.item():.6f} | "
                        f"Val Loss: {average_val_loss:.6f} | "
                        f"Validation Accuracy: {validation_accuracy:.2%}{schedule_suffix} | "
                        f"{format_peak_vram(device)}"
                    )

