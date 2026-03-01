"""
PPO (Proximal Policy Optimization) models.

Implements actor-critic architectures for PPO training.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.utils.initialization import orthogonal_init_


class PPOActor(BaseAgentModel):
    """
    Actor network for PPO.
    
    Outputs action probabilities for discrete actions or
    mean/std for continuous actions.
    
    Args:
        config: Model configuration
        continuous: Whether action space is continuous
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        continuous: bool = False,
    ):
        super().__init__(config)
        self.continuous = continuous
        
        # Backbone
        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        last_dim = config.hidden_dims[-1]
        
        if continuous:
            # Mean and log-std for continuous actions
            self.action_mean = nn.Linear(last_dim, config.action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(config.action_dim))
            orthogonal_init_(self.action_mean, gain=0.01)
        else:
            # Logits for discrete actions
            self.action_head = nn.Linear(last_dim, config.action_dim)
            orthogonal_init_(self.action_head, gain=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone."""
        return self.backbone(x)
    
    def get_distribution(self, observation: torch.Tensor):
        """Get action distribution."""
        features = self.forward(observation)
        
        if self.continuous:
            mean = self.action_mean(features)
            std = self.action_logstd.exp().expand_as(mean)
            return dist.Normal(mean, std)
        else:
            logits = self.action_head(features)
            return dist.Categorical(logits=logits)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        if deterministic:
            if self.continuous:
                features = self.forward(observation)
                return self.action_mean(features)
            else:
                features = self.forward(observation)
                logits = self.action_head(features)
                return torch.argmax(logits, dim=-1)
        else:
            d = self.get_distribution(observation)
            return d.sample()
    
    def get_log_prob(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probability of action."""
        d = self.get_distribution(observation)
        return d.log_prob(action).sum(dim=-1) if self.continuous else d.log_prob(action)
    
    def get_entropy(self, observation: torch.Tensor) -> torch.Tensor:
        """Get entropy of action distribution."""
        d = self.get_distribution(observation)
        return d.entropy().sum(dim=-1) if self.continuous else d.entropy()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Dummy value method (not used in actor-only)."""
        raise NotImplementedError("Use PPOActorCritic for value estimation")


class PPOCritic(BaseAgentModel):
    """
    Critic network for PPO.
    
    Estimates state value function V(s).
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        
        # Backbone
        layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        last_dim = config.hidden_dims[-1]
        
        # Value head
        self.value_head = nn.Linear(last_dim, 1)
        orthogonal_init_(self.value_head, gain=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone."""
        return self.backbone(x)
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self.forward(observation)
        return self.value_head(features).squeeze(-1)
    
    def get_action(self, *args, **kwargs) -> torch.Tensor:
        """Not implemented for critic."""
        raise NotImplementedError("Critic does not produce actions")


class PPOActorCritic(BaseAgentModel):
    """
    Combined Actor-Critic network for PPO.
    
    Shares backbone between actor and critic for efficiency.
    
    Args:
        config: Model configuration
        continuous: Whether action space is continuous
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
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        last_dim = config.hidden_dims[-1]
        
        # Actor head
        if continuous:
            self.action_mean = nn.Linear(last_dim, config.action_dim)
            self.action_logstd = nn.Parameter(torch.zeros(config.action_dim))
            orthogonal_init_(self.action_mean, gain=0.01)
        else:
            self.action_head = nn.Linear(last_dim, config.action_dim)
            orthogonal_init_(self.action_head, gain=1.0)
        
        # Critic head
        self.critic_head = nn.Linear(last_dim, 1)
        orthogonal_init_(self.critic_head, gain=1.0)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (action_output, value)
        """
        features = self.backbone(x)
        
        if self.continuous:
            action_out = self.action_mean(features)
        else:
            action_out = self.action_head(features)
        
        value = self.critic_head(features).squeeze(-1)
        
        return action_out, value
    
    def get_distribution(self, observation: torch.Tensor):
        """Get action distribution."""
        features = self.backbone(observation)
        
        if self.continuous:
            mean = self.action_mean(features)
            std = self.action_logstd.exp().expand_as(mean)
            return dist.Normal(mean, std)
        else:
            logits = self.action_head(features)
            return dist.Categorical(logits=logits)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        if deterministic:
            if self.continuous:
                features = self.backbone(observation)
                return self.action_mean(features)
            else:
                features = self.backbone(observation)
                logits = self.action_head(features)
                return torch.argmax(logits, dim=-1)
        else:
            d = self.get_distribution(observation)
            return d.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self.backbone(observation)
        return self.critic_head(features).squeeze(-1)
    
    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, and value in single forward pass.
        
        Returns:
            (action, log_prob, entropy, value)
        """
        features = self.backbone(observation)
        
        # Get distribution
        if self.continuous:
            mean = self.action_mean(features)
            std = self.action_logstd.exp().expand_as(mean)
            d = dist.Normal(mean, std)
        else:
            logits = self.action_head(features)
            d = dist.Categorical(logits=logits)
        
        # Sample or use provided action
        if action is None:
            if deterministic:
                action = d.mean if self.continuous else torch.argmax(logits, dim=-1)
            else:
                action = d.sample()
        
        log_prob = d.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        entropy = d.entropy()
        if self.continuous:
            entropy = entropy.sum(dim=-1)
        
        value = self.critic_head(features).squeeze(-1)
        
        return action, log_prob, entropy, value
    
    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for PPO update).
        
        Returns:
            (log_prob, entropy, value)
        """
        features = self.backbone(observation)
        
        if self.continuous:
            mean = self.action_mean(features)
            std = self.action_logstd.exp().expand_as(mean)
            d = dist.Normal(mean, std)
            log_prob = d.log_prob(action).sum(dim=-1)
        else:
            logits = self.action_head(features)
            d = dist.Categorical(logits=logits)
            log_prob = d.log_prob(action)
        
        entropy = d.entropy()
        if self.continuous:
            entropy = entropy.sum(dim=-1)
        
        value = self.critic_head(features).squeeze(-1)
        
        return log_prob, entropy, value
