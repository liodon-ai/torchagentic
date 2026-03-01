"""
PPO Training Example

This example demonstrates how to train a PPO agent.
"""

import torch
import torch.nn as nn
from typing import Tuple

from torchagentic import PPOActorCritic, ModelConfig


class RolloutBuffer:
    """Rollout buffer for PPO."""
    
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float,
            done: bool, log_prob: torch.Tensor, value: torch.Tensor):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        
        # Compute advantages
        advantages = self._compute_gae(rewards, values, dones)
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return states, actions, log_probs, advantages, returns
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                     dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)


def ppo_update(
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 64,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict:
    """
    Perform PPO update.
    
    Returns:
        Dictionary with loss statistics
    """
    dataset_size = states.shape[0]
    stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
    
    for _ in range(epochs):
        # Shuffle data
        indices = torch.randperm(dataset_size)
        
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            
            # Get new log probs and values
            log_probs, entropies, values = model.evaluate_actions(
                batch_states, batch_actions
            )
            
            # Policy loss
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = (values - batch_returns).pow(2).mean()
            
            # Entropy bonus
            entropy_bonus = entropies.mean()
            
            # Total loss
            loss = (
                policy_loss +
                value_coef * value_loss -
                entropy_coef * entropy_bonus
            )
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # Accumulate stats
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy_bonus.item()
    
    # Average stats
    num_updates = epochs * (dataset_size // batch_size)
    for key in stats:
        stats[key] /= num_updates
    
    return stats


def train_ppo(
    num_episodes: int = 500,
    steps_per_episode: int = 2048,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    epochs: int = 10,
    batch_size: int = 64,
):
    """
    Train PPO agent.
    
    Note: This example uses mock data for demonstration.
    Replace with actual environment for real training.
    """
    # Create model
    config = ModelConfig(
        input_dim=8,
        action_dim=2,
        hidden_dims=[64, 64],
    )
    model = PPOActorCritic(config=config, continuous=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer(gamma=gamma, lam=lam)
    
    print("Starting PPO training...")
    print(f"Model: {model}")
    
    for episode in range(num_episodes):
        buffer.reset()
        episode_reward = 0
        
        # Collect rollout
        state = torch.randn(8)  # Replace with env.reset()
        
        for step in range(steps_per_episode):
            # Get action
            with torch.no_grad():
                action, log_prob, entropy, value = model.get_action_and_value(
                    state.unsqueeze(0)
                )
            
            # Step environment (mock)
            next_state = torch.randn(8)  # Replace with env.step()
            reward = 1.0  # Replace with actual reward
            done = step >= steps_per_episode - 1
            
            # Store transition
            buffer.add(state, action, reward, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # PPO update
        states, actions, old_log_probs, advantages, returns = buffer.get()
        
        stats = ppo_update(
            model=model,
            optimizer=optimizer,
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        # Log
        if episode % 50 == 0:
            print(
                f"Episode {episode}: "
                f"reward={episode_reward:.1f}, "
                f"policy_loss={stats['policy_loss']:.4f}, "
                f"value_loss={stats['value_loss']:.4f}"
            )
    
    print("Training completed!")
    return model


if __name__ == "__main__":
    model = train_ppo(num_episodes=200)
    
    # Save model
    model.save("ppo_model.pt")
    print("Model saved to ppo_model.pt")
