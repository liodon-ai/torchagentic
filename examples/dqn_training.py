"""
DQN Training Example

This example demonstrates how to train a DQN agent on CartPole.
"""

import torch
import torch.nn.functional as F
from collections import deque
import random

from torchagentic import DQN, ModelConfig


class ReplayBuffer:
    """Simple replay buffer for experience replay."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.buffer)


def train_dqn(
    num_episodes: int = 500,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    target_update: int = 10,
    lr: float = 1e-4,
    buffer_size: int = 10000,
):
    """
    Train DQN on CartPole environment.
    
    Note: This example uses a mock environment for demonstration.
    Replace with gymnasium.make('CartPole-v1') for actual training.
    """
    # Create models
    config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[128, 128])
    policy_net = DQN(config=config, image_input=False)
    target_net = DQN(config=config, image_input=False)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_size)
    
    epsilon = epsilon_start
    episode_rewards = []
    
    print("Starting DQN training...")
    print(f"Policy network: {policy_net}")
    
    for episode in range(num_episodes):
        # Reset environment (mock)
        state = torch.randn(4)  # Replace with env.reset()
        episode_reward = 0
        
        for t in range(500):  # Max episode length
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    action = policy_net.get_action(state.unsqueeze(0), deterministic=True).item()
            
            # Step environment (mock)
            next_state = torch.randn(4)  # Replace with env.step()
            reward = 1.0  # Replace with actual reward
            done = t >= 499  # Replace with actual done
            
            # Store transition
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Current Q values
                q_values = policy_net(states)
                q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                
                # Target Q values
                with torch.no_grad():
                    next_q_values = target_net(next_states)
                    next_q_max = next_q_values.max(1)[0]
                    targets = rewards + gamma * next_q_max * (1 - dones)
                
                # Loss
                loss = F.mse_loss(q_values, targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Log
        if episode % 50 == 0:
            avg_reward = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
            print(f"Episode {episode}: avg_reward={avg_reward:.1f}, epsilon={epsilon:.3f}")
    
    print("Training completed!")
    return policy_net


if __name__ == "__main__":
    model = train_dqn(num_episodes=200)
    
    # Save model
    model.save("dqn_cartpole.pt")
    print("Model saved to dqn_cartpole.pt")
