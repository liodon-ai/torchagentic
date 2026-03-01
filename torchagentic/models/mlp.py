"""
Multi-Layer Perceptron (MLP) networks for agents.

Provides flexible feedforward neural network architectures
that can be used as backbones for various agent types.
"""

from typing import Optional
import torch
import torch.nn as nn

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.utils.initialization import orthogonal_init_


class MLPNetwork(BaseAgentModel):
    """
    Flexible Multi-Layer Perceptron for agent models.
    
    Features:
    - Configurable hidden layers
    - Multiple activation functions
    - Optional dropout and normalization
    - Orthogonal initialization
    
    Args:
        config: Model configuration
    
    Example:
        >>> config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[128, 64])
        >>> mlp = MLPNetwork(config)
        >>> x = torch.randn(32, 10)
        >>> output = mlp(x)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        
        # Build layers
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(config.get_activation())
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Action head
        self.action_head = nn.Linear(config.hidden_dims[-1], config.action_dim)
        orthogonal_init_(self.action_head, gain=1.0)
        
        # Value head
        self.value_head = nn.Linear(config.hidden_dims[-1], 1)
        orthogonal_init_(self.value_head, gain=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone."""
        return self.backbone(x)
    
    def get_action_logits(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Get action logits for discrete action spaces."""
        features = self.forward(observation)
        return self.action_head(features)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action (discrete)."""
        logits = self.get_action_logits(observation)
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self.forward(observation)
        return self.value_head(features).squeeze(-1)
    
    def get_action_and_value(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get both action and value in single forward pass."""
        features = self.forward(observation)
        logits = self.action_head(features)
        value = self.value_head(features).squeeze(-1)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        
        return action, value


class ActorCriticMLP(BaseAgentModel):
    """
    Actor-Critic MLP with separate heads.
    
    Useful for policy gradient methods like PPO, A3C.
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        continuous: bool = False,
    ):
        super().__init__(config)
        self.continuous = continuous
        
        # Shared backbone
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(config.get_activation())
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        last_dim = config.hidden_dims[-1]
        
        # Actor head
        if continuous:
            self.action_mean = nn.Linear(last_dim, config.action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(config.action_dim))
            orthogonal_init_(self.action_mean, gain=0.01)
        else:
            self.actor = nn.Linear(last_dim, config.action_dim)
            orthogonal_init_(self.actor, gain=1.0)
        
        # Critic head
        self.critic = nn.Linear(last_dim, 1)
        orthogonal_init_(self.critic, gain=1.0)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning actor and critic outputs."""
        features = self.backbone(x)
        
        if self.continuous:
            action_mean = self.action_mean(features)
            action_logstd = self.action_logstd.expand_as(action_mean)
            actor_out = (action_mean, action_logstd)
        else:
            actor_out = self.actor(features)
        
        critic_out = self.critic(features).squeeze(-1)
        
        return actor_out, critic_out
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action from actor."""
        features = self.backbone(observation)
        
        if self.continuous:
            mean = self.action_mean(features)
            std = self.action_logstd.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            
            if deterministic:
                return mean
            return dist.sample()
        else:
            logits = self.actor(features)
            if deterministic:
                return torch.argmax(logits, dim=-1)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value from critic."""
        features = self.backbone(observation)
        return self.critic(features).squeeze(-1)
    
    def get_log_prob(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probability of action."""
        features = self.backbone(observation)
        
        if self.continuous:
            mean = self.action_mean(features)
            std = self.action_logstd.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
        else:
            logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=logits)
        
        return dist.log_prob(action)
    
    def get_entropy(self, observation: torch.Tensor) -> torch.Tensor:
        """Get entropy of action distribution."""
        features = self.backbone(observation)
        
        if self.continuous:
            mean = self.action_mean(features)
            std = self.action_logstd.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
        else:
            logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=logits)
        
        return dist.entropy()
