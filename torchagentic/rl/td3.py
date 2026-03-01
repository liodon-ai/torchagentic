"""
TD3 (Twin Delayed DDPG) models for continuous control.

Implements actor and critic architectures for TD3 training.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.utils.initialization import orthogonal_init_


class TD3Actor(BaseAgentModel):
    """
    Deterministic actor for TD3.
    
    Reference: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", ICML 2018.
    
    Args:
        config: Model configuration
        action_bounds: Tuple of (min, max) action values
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        action_bounds: Optional[Tuple[float, float]] = None,
    ):
        super().__init__(config)
        
        self.action_bounds = action_bounds or (-1.0, 1.0)
        
        # Backbone
        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        last_dim = config.hidden_dims[-1]
        
        # Action head with tanh
        self.action_head = nn.Linear(last_dim, config.action_dim)
        nn.init.uniform_(self.action_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.action_head.bias, -3e-3, 3e-3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        return torch.tanh(self.action_head(features))
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = True,
        noise: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Get action.
        
        Args:
            observation: Current observation
            deterministic: Use deterministic action
            noise: Optional noise to add for exploration
        """
        action = self.forward(observation)
        
        # Add noise if specified
        if noise is not None:
            action = action + torch.randn_like(action) * noise
            action = torch.clamp(action, -1, 1)
        
        # Scale to action bounds
        low, high = self.action_bounds
        action = action * (high - low) / 2 + (high + low) / 2
        
        return action
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Not applicable."""
        raise NotImplementedError("Actor does not estimate value")


class TD3Critic(nn.Module):
    """
    Q-function critic for TD3.
    
    Takes both observation and action as input.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        # Q-network
        layers = []
        prev_dim = input_dim + action_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:
                layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.q_network = nn.Sequential(*layers)
        orthogonal_init_(self.q_network[-1], gain=1.0)
    
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([observation, action], dim=-1)
        return self.q_network(x).squeeze(-1)
    
    def get_q_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-value."""
        return self.forward(observation, action)


class TD3ActorCritic(nn.Module):
    """
    Combined Actor-Critic for TD3.
    
    Includes actor and two Q-critics for double Q-learning.
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        action_bounds: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        
        self.actor = TD3Actor(config, action_bounds)
        
        # Two critics for double Q-learning
        self.q1 = TD3Critic(
            config.input_dim,
            config.action_dim,
            config.hidden_dims,
        )
        self.q2 = TD3Critic(
            config.input_dim,
            config.action_dim,
            config.hidden_dims,
        )
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = True,
        noise: Optional[float] = None,
    ) -> torch.Tensor:
        """Get action."""
        return self.actor.get_action(observation, deterministic, noise)
    
    def get_q_values(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both critics."""
        return self.q1(observation, action), self.q2(observation, action)
    
    def get_min_q(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get minimum Q-value."""
        q1, q2 = self.get_q_values(observation, action)
        return torch.min(q1, q2)
