"""
SAC (Soft Actor-Critic) models for continuous control.

Implements actor and critic architectures for SAC training.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.utils.initialization import orthogonal_init_


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACActor(BaseAgentModel):
    """
    Stochastic actor for SAC.
    
    Outputs a Gaussian distribution over continuous actions.
    Uses squashed Gaussian for bounded action spaces.
    
    Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL", ICML 2018.
    
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
        
        # Mean and log-std heads
        self.mean_head = nn.Linear(last_dim, config.action_dim)
        self.logstd_head = nn.Linear(last_dim, config.action_dim)
        
        orthogonal_init_(self.mean_head, gain=0.01)
        orthogonal_init_(self.logstd_head, gain=0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (mean, log_std)
        """
        features = self.backbone(x)
        mean = self.mean_head(features)
        log_std = self.logstd_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def get_distribution(self, observation: torch.Tensor) -> dist.TransformedDistribution:
        """
        Get squashed Gaussian distribution.
        
        Returns:
            TransformedDistribution (squashed Gaussian)
        """
        mean, log_std = self.forward(observation)
        std = log_std.exp()
        
        base_dist = dist.Normal(mean, std)
        
        # Squash with tanh
        return dist.TransformedDistribution(
            base_dist,
            [dist.transforms.TanhTransform(cache_size=1)],
        )
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action.
        
        Args:
            observation: Current observation
            deterministic: Use mean action
            with_log_prob: Also return log probability
        
        Returns:
            action, (log_prob if with_log_prob)
        """
        if deterministic:
            mean, _ = self.forward(observation)
            action = torch.tanh(mean)
            
            # Scale to action bounds
            low, high = self.action_bounds
            action = action * (high - low) / 2 + (high + low) / 2
            
            if with_log_prob:
                return action, torch.zeros(observation.size(0), device=observation.device)
            return action
        
        dist = self.get_distribution(observation)
        action = dist.rsample()  # Reparameterized sample
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Scale to action bounds
        low, high = self.action_bounds
        action = action * (high - low) / 2 + (high + low) / 2
        
        if with_log_prob:
            return action, log_prob
        return action
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Not applicable for actor."""
        raise NotImplementedError("Actor does not estimate value")


class SACCritic(nn.Module):
    """
    Q-function critic for SAC.
    
    Takes both observation and action as input.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Q-network
        layers = []
        prev_dim = input_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.q_network = nn.Sequential(*layers)
        orthogonal_init_(self.q_network[-1])
    
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning Q-value."""
        x = torch.cat([observation, action], dim=-1)
        return self.q_network(x).squeeze(-1)
    
    def get_q_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-value for state-action pair."""
        return self.forward(observation, action)


class SACValue(BaseAgentModel):
    """
    State value function for SAC (V-function).
    
    Estimates V(s) = E[Q(s, a) - log pi(a|s)]
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        
        # Backbone
        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        last_dim = config.hidden_dims[-1]
        
        # Value head
        self.value_head = nn.Linear(last_dim, 1)
        orthogonal_init_(self.value_head)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        return self.value_head(features).squeeze(-1)
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        return self.forward(observation)
    
    def get_action(self, *args, **kwargs) -> torch.Tensor:
        """Not applicable."""
        raise NotImplementedError("Value function does not produce actions")


class SACActorCritic(nn.Module):
    """
    Combined Actor-Critic for SAC.
    
    Includes actor and two Q-critics for double Q-learning.
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        action_bounds: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        
        self.actor = SACActor(config, action_bounds)
        
        # Two Q-critics for double Q-learning
        self.q1 = SACCritic(
            config.input_dim,
            config.action_dim,
            config.hidden_dims,
        )
        self.q2 = SACCritic(
            config.input_dim,
            config.action_dim,
            config.hidden_dims,
        )
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ):
        """Get action from actor."""
        return self.actor.get_action(observation, deterministic, with_log_prob)
    
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
        """Get minimum Q-value (for clipping)."""
        q1, q2 = self.get_q_values(observation, action)
        return torch.min(q1, q2)
