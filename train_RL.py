import torch

from gpu_cleanup import format_peak_vram
from metrics import epoch_metrics, evaluate_classifier, training_evaluation_loader

def longest_streak_reward(actions, labels):

    correct = actions.eq(labels)

    current = torch.zeros_like(actions[:, 0])
    longest = current.clone()

    for t in range(actions.size(1)):

        current = (current + 1) * correct[:, t]

        longest = torch.maximum(longest, current)

    return longest.float()


def rl_train(
    model, train_loader, optimizer, epochs, device, print_interval=1,
    valid_loader=None, history=None, epoch_loader=None, augment_batch=None,
    entropy_coefficient=0.01
):

    model.to(device)
    model.train()
    reward_baseline = 0.0
    train_eval_loader = training_evaluation_loader(train_loader) if history is not None else None

    for epoch in range(1, epochs + 1):

        total_reward = 0
        total_correct = 0
        total_samples = 0
        total_episodes = 0

        batches = epoch_loader(epoch) if epoch_loader is not None else train_loader
        for inputs, labels in batches:

            inputs = inputs.to(device)
            labels = labels.to(device)
            if augment_batch is not None:
                inputs = augment_batch(inputs)

            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)

            # Create policy distribution
            policy = torch.distributions.Categorical(logits=logits)

            # Sample classifications
            actions = policy.sample()
            correct = actions.eq(labels)

            # Calculate reward for each episode
            correct_counts = correct.sum(dim=1)
            rewards = correct_counts + longest_streak_reward(actions, labels)

            # Track total reward and accuracy
            batch_reward = rewards.sum().item()
            total_reward += batch_reward
            total_episodes += rewards.numel()

            total_correct += correct_counts.sum().item()
            total_samples += labels.numel()

            # Weight sampled actions using the reward and the running baseline.
            log_probs = policy.log_prob(actions).sum(dim=1)
            advantages = rewards - reward_baseline
            policy_loss = -(advantages * log_probs).mean()
            entropy = policy.entropy().mean()
            loss = policy_loss - entropy_coefficient * entropy
            reward_baseline = 0.9 * reward_baseline + 0.1 * (batch_reward / rewards.numel())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

        if history is not None:
            # Iterating a DataLoader draws a CPU seed even when it does not shuffle.
            with torch.random.fork_rng(devices=[]):
                targets, predictions, _ = evaluate_classifier(model, valid_loader, device)
            history.append(epoch_metrics(
                epoch, model, train_eval_loader, device, targets, predictions
            ))

        if total_episodes and (epoch % print_interval == 0 or epoch == epochs):
            validation = (
                f" | Validation Accuracy: {history[-1].validation_accuracy:.2%}"
                if history else ""
            )
            print(
                f"Epoch: {epoch}/{epochs} | "
                f"Average Reward: {total_reward / total_episodes:.4f} | "
                f"Training Sampled Accuracy: "
                f"{total_correct / total_samples:.2%}{validation} | {format_peak_vram(device)}"
            )

