"""
Decision Transformer Example

This example demonstrates offline RL with Decision Transformer.
"""

import torch
from torchagentic import DecisionTransformer, ModelConfig


class TrajectoryDataset:
    """Dataset for offline RL trajectories."""
    
    def __init__(self, trajectories: list, max_len: int = 20):
        """
        Args:
            trajectories: List of trajectories, each containing
                         (states, actions, rewards, dones)
            max_len: Maximum sequence length
        """
        self.trajectories = trajectories
        self.max_len = max_len
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        states, actions, rewards, dones = traj
        
        # Compute returns-to-go
        returns_to_go = []
        cumulative = 0
        for r in reversed(rewards):
            cumulative += r
            returns_to_go.insert(0, cumulative)
        
        # Truncate or pad
        T = min(len(states), self.max_len)
        
        states = states[:T]
        actions = actions[:T]
        returns_to_go = returns_to_go[:T]
        
        # Pad if needed
        if T < self.max_len:
            pad_state = torch.zeros_like(states[0])
            pad_action = torch.zeros_like(actions[0])
            
            states = states + [pad_state] * (self.max_len - T)
            actions = actions + [pad_action] * (self.max_len - T)
            returns_to_go = returns_to_go + [0] * (self.max_len - T)
        
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(returns_to_go).unsqueeze(-1),
        )


def train_decision_transformer(
    num_episodes: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    max_seq_len: int = 20,
):
    """
    Train Decision Transformer for offline RL.
    """
    # Create model
    config = ModelConfig(input_dim=17, action_dim=3)  # CartPole example
    model = DecisionTransformer(
        config=config,
        embed_dim=128,
        num_heads=1,
        num_layers=3,
        max_seq_len=max_seq_len,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create mock dataset (replace with actual offline dataset)
    trajectories = []
    for _ in range(1000):
        states = [torch.randn(17) for _ in range(50)]
        actions = [torch.randn(3) for _ in range(50)]
        rewards = [1.0 for _ in range(50)]
        trajectories.append((states, actions, rewards, [False] * 50))
    
    dataset = TrajectoryDataset(trajectories, max_len=max_seq_len)
    
    print("Starting Decision Transformer training...")
    print(f"Model: {model}")
    print(f"Dataset size: {len(dataset)} trajectories")
    
    for episode in range(num_episodes):
        total_loss = 0
        num_batches = 0
        
        # Training loop
        indices = torch.randperm(len(dataset))
        
        for start in range(0, len(dataset), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch_states = []
            batch_actions = []
            batch_rtgs = []
            
            for idx in batch_indices:
                s, a, r = dataset[idx]
                batch_states.append(s)
                batch_actions.append(a)
                batch_rtgs.append(r)
            
            batch_states = torch.stack(batch_states)
            batch_actions = torch.stack(batch_actions)
            batch_rtgs = torch.stack(batch_rtgs)
            
            # Forward pass
            predicted_actions = model(batch_states, batch_actions, batch_rtgs)
            
            # Loss: MSE between predicted and actual actions
            # Only compute loss for valid timesteps
            T = batch_actions.shape[1]
            loss = F.mse_loss(predicted_actions, batch_actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        if episode % 20 == 0:
            print(f"Episode {episode}: loss={avg_loss:.4f}")
    
    print("Training completed!")
    return model


def evaluate_decision_transformer(
    model: DecisionTransformer,
    num_episodes: int = 10,
):
    """Evaluate trained Decision Transformer."""
    model.eval()
    
    total_return = 0
    
    for _ in range(num_episodes):
        # Initialize
        state = torch.randn(17)  # Replace with env.reset()
        target_return = 100  # Desired return
        
        episode_return = 0
        past_states = []
        past_actions = []
        
        for t in range(100):
            # Compute return-to-go
            rtg = target_return - episode_return
            
            # Get action
            with torch.no_grad():
                if len(past_actions) == 0:
                    action = model.get_action(
                        state.unsqueeze(0),
                        torch.tensor([[rtg]]),
                        None,
                        deterministic=True,
                    )
                else:
                    past_states_tensor = torch.stack(past_states[-model.max_seq_len:])
                    past_actions_tensor = torch.stack(past_actions[-model.max_seq_len:])
                    
                    action = model.get_action(
                        state.unsqueeze(0),
                        torch.tensor([[rtg]]),
                        past_actions_tensor.unsqueeze(0),
                        deterministic=True,
                    )
            
            # Step environment (mock)
            next_state = torch.randn(17)
            reward = 1.0
            
            episode_return += reward
            past_states.append(state)
            past_actions.append(action.squeeze(0))
            
            state = next_state
        
        total_return += episode_return
    
    avg_return = total_return / num_episodes
    print(f"Average return: {avg_return:.1f}")
    
    return avg_return


if __name__ == "__main__":
    import torch.nn.functional as F
    
    model = train_decision_transformer(num_episodes=100)
    
    # Save model
    model.save("decision_transformer.pt")
    print("Model saved to decision_transformer.pt")
    
    # Evaluate
    evaluate_decision_transformer(model)
