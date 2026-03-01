"""
Multi-Agent Training Example (MADDPG)

This example demonstrates how to train multi-agent systems.
"""

import torch
import torch.nn.functional as F
from typing import List

from torchagentic import MADDPGAgent


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent training."""
    
    def __init__(self, num_agents: int, capacity: int = 10000):
        self.num_agents = num_agents
        self.buffer = []
        self.capacity = capacity
        self.idx = 0
    
    def push(self, states, actions, rewards, next_states, dones):
        """
        Store transition.
        
        Args:
            states: List of states for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_states: List of next states for each agent
            dones: List of done flags for each agent
        """
        transition = (states, actions, rewards, next_states, dones)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        
        self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size: int):
        import random
        batch = random.sample(self.buffer, batch_size)
        
        # Stack each component
        states = [torch.stack([t[0][i] for t in batch]) for i in range(self.num_agents)]
        actions = [torch.stack([t[1][i] for t in batch]) for i in range(self.num_agents)]
        rewards = [torch.tensor([t[2][i] for t in batch], dtype=torch.float32)
                   for i in range(self.num_agents)]
        next_states = [torch.stack([t[3][i] for t in batch]) for i in range(self.num_agents)]
        dones = [torch.tensor([t[4][i] for t in batch], dtype=torch.float32)
                 for i in range(self.num_agents)]
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


def train_maddpg(
    num_agents: int = 3,
    num_episodes: int = 500,
    batch_size: int = 128,
    gamma: float = 0.95,
    tau: float = 0.01,
    lr: float = 1e-3,
    buffer_size: int = 10000,
):
    """
    Train MADDPG agents.
    
    Note: This example uses mock data for demonstration.
    """
    # Create model
    model = MADDPGAgent(
        num_agents=num_agents,
        obs_dim=10,
        action_dim=2,
        hidden_dims=[256, 256],
        shared_params=True,
    )
    
    # Create target networks
    target_model = MADDPGAgent(
        num_agents=num_agents,
        obs_dim=10,
        action_dim=2,
        hidden_dims=[256, 256],
        shared_params=True,
    )
    target_model.load_state_dict(model.state_dict())
    
    # Optimizers
    actor_optimizer = torch.optim.Adam(model.actors.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=lr)
    
    buffer = MultiAgentReplayBuffer(num_agents=num_agents, capacity=buffer_size)
    
    print("Starting MADDPG training...")
    print(f"Model: {model}")
    
    for episode in range(num_episodes):
        episode_rewards = [0.0] * num_agents
        
        # Reset (mock)
        states = [torch.randn(10) for _ in range(num_agents)]
        
        for step in range(100):  # Episode length
            # Get actions
            states_tensor = torch.stack(states).unsqueeze(0)  # (1, num_agents, obs_dim)
            actions = model.get_actions(states_tensor, noise=0.1)
            actions_list = [actions[0, i] for i in range(num_agents)]
            
            # Step environment (mock)
            next_states = [torch.randn(10) for _ in range(num_agents)]
            rewards = [1.0 for _ in range(num_agents)]
            dones = [step >= 99 for _ in range(num_agents)]
            
            # Store transition
            buffer.push(states, actions_list, rewards, next_states, dones)
            
            for i in range(num_agents):
                episode_rewards[i] += rewards[i]
            
            states = next_states
            
            if all(dones):
                break
            
            # Train if enough samples
            if len(buffer) >= batch_size:
                # Sample batch
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                    buffer.sample(batch_size)
                
                # Stack for centralized critic
                batch_states_all = torch.stack(batch_states, dim=1)  # (batch, num_agents, obs_dim)
                batch_actions_all = torch.stack(batch_actions, dim=1)  # (batch, num_agents, action_dim)
                batch_next_states_all = torch.stack(batch_next_states, dim=1)
                
                # Critic update
                with torch.no_grad():
                    # Target actions
                    target_actions = target_model.get_actions(batch_next_states_all, deterministic=True)
                    
                    # Target Q values
                    target_q = target_model.get_q_value(batch_next_states_all, target_actions)
                    
                    # Compute targets
                    targets = torch.stack(batch_rewards).mean(0) + gamma * target_q * \
                              (1 - torch.stack(batch_dones).mean(0))
                
                # Current Q values
                current_q = model.get_q_value(batch_states_all, batch_actions_all)
                
                # Critic loss
                critic_loss = F.mse_loss(current_q, targets)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Actor update (for each agent)
                for agent_id in range(num_agents):
                    actor = model.get_actor(agent_id)
                    
                    # Get actions for this agent
                    agent_states = batch_states[agent_id]
                    agent_actions = actor(agent_states)
                    
                    # Replace actions in batch
                    actions_copy = batch_actions_all.clone()
                    actions_copy[:, agent_id, :] = agent_actions
                    
                    # Compute Q with new actions
                    q_new = model.get_q_value(batch_states_all, actions_copy)
                    
                    # Actor loss (maximize Q)
                    actor_loss = -q_new.mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                
                # Update target networks
                for target_param, param in zip(target_model.parameters(), model.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Log
        if episode % 50 == 0:
            avg_reward = sum(episode_rewards) / num_agents
            print(f"Episode {episode}: avg_reward={avg_reward:.2f}")
    
    print("Training completed!")
    return model


if __name__ == "__main__":
    model = train_maddpg(num_agents=3, num_episodes=200)
    
    # Save model
    torch.save(model.state_dict(), "maddpg_model.pt")
    print("Model saved to maddpg_model.pt")
