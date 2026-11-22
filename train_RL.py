import torch

def longest_streak_reward(actions, labels):

    correct = actions.eq(labels)

    current = torch.zeros_like(actions[:, 0])
    longest = current.clone()

    for t in range(actions.size(1)):

        current = (current + 1) * correct[:, t]

        longest = torch.maximum(longest, current)

    return longest.float()


def rl_train(model, train_loader, optimizer, epochs, device):

    model.to(device)
    model.train()

    for epoch in range(epochs):

        total_reward = 0
        total_correct = 0
        total_samples = 0
        total_episodes = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)

            # Create policy distribution
            policy = torch.distributions.Categorical(logits=logits)

            # Sample classifications
            actions = policy.sample()
            correct = actions.eq(labels)

            # Calculate reward for each episode
            rewards = longest_streak_reward(actions, labels)

            # Track total reward and accuracy
            total_reward += rewards.sum().item()
            total_episodes += rewards.numel()

            total_correct += correct.sum().item()
            total_samples += labels.numel()

            # Flip one prediction at a time to measure its effect on the streak.
            positions = torch.eye(labels.size(1), dtype=torch.bool, device=labels.device)
            forced_correct = (correct.unsqueeze(1) | positions).flatten(0, 1).long()
            forced_wrong = (correct.unsqueeze(1) & ~positions).flatten(0, 1).long()
            targets = torch.ones_like(forced_correct)
            reward_if_correct = longest_streak_reward(forced_correct, targets)
            reward_if_wrong = longest_streak_reward(forced_wrong, targets)
            credit = (reward_if_correct - reward_if_wrong).reshape_as(labels)
            correct_probs = policy.probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            policy_loss = -(credit * correct_probs).sum(dim=1).mean()
            entropy = policy.entropy().mean()
            loss = policy_loss - 0.01 * entropy

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

        if total_episodes:
            print(
                f"Epoch: {epoch + 1}/{epochs} | "
                f"Average Reward: {total_reward / total_episodes:.4f} | "
                f"Training Accuracy: "
                f"{total_correct / total_samples:.2%}"
            )

