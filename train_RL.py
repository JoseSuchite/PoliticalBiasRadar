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
    reward_baseline = 0.0

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

            # Record action log probabilities
            log_probs = policy.log_prob(actions)

            # Calculate reward for each episode
            rewards = longest_streak_reward(actions, labels)

            # Track total reward and accuracy
            total_reward += rewards.sum().item()
            total_episodes += rewards.numel()

            total_correct += actions.eq(labels).sum().item()
            total_samples += labels.numel()

            # REINFORCE policy gradient loss and reduce variance with a baseline

            entropy = policy.entropy().mean()

            advantages = rewards - reward_baseline

            policy_loss = -(advantages * log_probs.sum(dim=1)).mean()

            loss =  policy_loss - 0.01 * entropy

            reward_baseline = (0.9 * reward_baseline + 0.1 * rewards.mean().item())

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

