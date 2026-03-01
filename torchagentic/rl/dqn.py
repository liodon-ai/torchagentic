"""
DQN (Deep Q-Network) models for reinforcement learning.

Implements various DQN architectures including:
- Standard DQN
- Dueling DQN
- Noisy DQN
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.utils.initialization import orthogonal_init_


class DQN(BaseAgentModel):
    """
    Standard Deep Q-Network.
    
    Reference: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015.
    
    Args:
        config: Model configuration
        image_input: Whether input is image (CHW) or vector
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_input: bool = False,
    ):
        super().__init__(config)
        
        self.image_input = image_input
        
        if image_input:
            # CNN backbone for image input
            self.backbone = nn.Sequential(
                nn.Conv2d(config.input_dim, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
            )
            
            # Calculate flat dim
            with torch.no_grad():
                dummy = torch.zeros(1, config.input_dim, 84, 84)
                flat_dim = self.backbone(dummy).view(1, -1).shape[1]
        else:
            # MLP backbone for vector input
            layers = []
            prev_dim = config.input_dim
            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.backbone = nn.Sequential(*layers)
            flat_dim = config.hidden_dims[-1]
        
        # Q-value head
        self.q_head = nn.Linear(flat_dim, config.action_dim)
        orthogonal_init_(self.q_head)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values."""
        if self.image_input and x.dim() == 3:
            x = x.transpose(1, 3)
        
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.q_head(features)
    
    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions."""
        return self.forward(observation)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            deterministic: If True, ignore epsilon and take argmax
            epsilon: Exploration rate for epsilon-greedy
        """
        q_values = self.get_q_values(observation)
        
        if deterministic:
            return torch.argmax(q_values, dim=-1)
        
        # Epsilon-greedy
        batch_size = q_values.shape[0]
        random_actions = torch.randint(
            0, self.config.action_dim, (batch_size,),
            device=q_values.device,
        )
        greedy_actions = torch.argmax(q_values, dim=-1)
        
        # Random mask
        random_mask = torch.rand(batch_size, device=q_values.device) < epsilon
        return torch.where(random_mask, random_actions, greedy_actions)
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value (max Q-value)."""
        q_values = self.get_q_values(observation)
        return torch.max(q_values, dim=-1).values


class DuelingDQN(BaseAgentModel):
    """
    Dueling DQN architecture.
    
    Separates value and advantage streams for better Q-value estimation.
    
    Reference: Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016.
    
    Args:
        config: Model configuration
        image_input: Whether input is image
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_input: bool = False,
    ):
        super().__init__(config)
        
        self.image_input = image_input
        
        if image_input:
            self.backbone = nn.Sequential(
                nn.Conv2d(config.input_dim, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
            )
            
            with torch.no_grad():
                dummy = torch.zeros(1, config.input_dim, 84, 84)
                flat_dim = self.backbone(dummy).view(1, -1).shape[1]
        else:
            layers = []
            prev_dim = config.input_dim
            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.backbone = nn.Sequential(*layers)
            flat_dim = config.hidden_dims[-1]
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(flat_dim, flat_dim // 2),
            nn.ReLU(),
            nn.Linear(flat_dim // 2, 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_dim, flat_dim // 2),
            nn.ReLU(),
            nn.Linear(flat_dim // 2, config.action_dim),
        )
        
        orthogonal_init_(self.value_stream[-1])
        orthogonal_init_(self.advantage_stream[-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values."""
        if self.image_input and x.dim() == 3:
            x = x.transpose(1, 3)
        
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Value and advantage
        value = self.value_stream(features)  # (B, 1)
        advantage = self.advantage_stream(features)  # (B, A)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values
    
    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor:
        """Get Q-values."""
        return self.forward(observation)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Get action."""
        q_values = self.get_q_values(observation)
        
        if deterministic:
            return torch.argmax(q_values, dim=-1)
        
        batch_size = q_values.shape[0]
        random_actions = torch.randint(
            0, self.config.action_dim, (batch_size,),
            device=q_values.device,
        )
        greedy_actions = torch.argmax(q_values, dim=-1)
        
        random_mask = torch.rand(batch_size, device=q_values.device) < epsilon
        return torch.where(random_mask, random_actions, greedy_actions)
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        if self.image_input and observation.dim() == 3:
            obs = observation.transpose(1, 3)
        else:
            obs = observation
        
        features = self.backbone(obs).view(observation.size(0), -1)
        return self.value_stream(features).squeeze(-1)


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration.
    
    Reference: Fortunato et al., "Noisy Networks for Exploration", ICLR 2018.
    """
    
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.zeros(out_features))
        
        # Reset parameters
        self.reset_parameters()
        
        # Noise buffers
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
    
    def reset_parameters(self) -> None:
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))
    
    def _sample_noise(self, size: torch.Size) -> torch.Tensor:
        """Sample noise from factorized Gaussian."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with noise."""
        if self.training:
            # Sample noise
            eps_in = self._sample_noise((self.in_features,))
            eps_out = self._sample_noise((self.out_features,))
            eps = eps_out.ger(eps_in)  # Outer product
            
            self.weight_epsilon = eps
            self.bias_epsilon = eps_out
        else:
            self.weight_epsilon.zero_()
            self.bias_epsilon.zero_()
        
        # Compute noisy weights
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        
        return F.linear(x, weight, bias)


class NoisyDQN(BaseAgentModel):
    """
    DQN with Noisy Networks for exploration.
    
    Uses parameterized noise instead of epsilon-greedy for exploration.
    
    Reference: Fortunato et al., "Noisy Networks for Exploration", ICLR 2018.
    
    Args:
        config: Model configuration
        image_input: Whether input is image
        sigma0: Initial noise standard deviation
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_input: bool = False,
        sigma0: float = 0.5,
    ):
        super().__init__(config)
        
        self.image_input = image_input
        
        if image_input:
            self.backbone = nn.Sequential(
                nn.Conv2d(config.input_dim, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
            )
            
            with torch.no_grad():
                dummy = torch.zeros(1, config.input_dim, 84, 84)
                flat_dim = self.backbone(dummy).view(1, -1).shape[1]
        else:
            layers = []
            prev_dim = config.input_dim
            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            self.backbone = nn.Sequential(*layers)
            flat_dim = config.hidden_dims[-1]
        
        # Noisy linear layer
        self.q_head = NoisyLinear(flat_dim, config.action_dim, sigma0=sigma0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.image_input and x.dim() == 3:
            x = x.transpose(1, 3)
        
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.q_head(features)
    
    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor:
        """Get Q-values."""
        return self.forward(observation)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get action.
        
        In training mode, noise is automatically added.
        In eval mode, use deterministic=True for greedy actions.
        """
        training = self.training
        
        if deterministic:
            self.eval()
        
        with torch.no_grad():
            q_values = self.get_q_values(observation)
            action = torch.argmax(q_values, dim=-1)
        
        if training:
            self.train()
        
        return action
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value (max Q-value)."""
        q_values = self.get_q_values(observation)
        return torch.max(q_values, dim=-1).values
    
    def reset_noise(self) -> None:
        """Reset noise in noisy layers (called at each episode)."""
        # Noise is automatically sampled in forward pass during training
        pass
