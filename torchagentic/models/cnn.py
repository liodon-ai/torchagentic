"""
Convolutional Neural Networks for visual agents.

Provides CNN architectures for processing image observations,
including Nature DQN CNN and ResNet variants.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.utils.initialization import orthogonal_init_


class CNNNetwork(BaseAgentModel):
    """
    Generic CNN for image-based agents.
    
    Args:
        config: Model configuration (input_dim used as image channels)
        image_shape: Shape of input image (C, H, W)
        features_dim: Output feature dimension after CNN
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_shape: Tuple[int, int, int] = (3, 84, 84),
        features_dim: int = 512,
    ):
        super().__init__(config)
        
        self.image_shape = image_shape
        self.channels, self.height, self.width = image_shape
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            conv_out = self.conv_layers(dummy)
            flat_dim = conv_out.view(1, -1).shape[1]
        
        self.features_dim = features_dim
        self.fc = nn.Linear(flat_dim, features_dim)
        nn.init.relu_(self.fc.weight)
        
        # Action and value heads
        self.action_head = nn.Linear(features_dim, config.action_dim)
        orthogonal_init_(self.action_head)
        
        self.value_head = nn.Linear(features_dim, 1)
        orthogonal_init_(self.value_head)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through CNN."""
        # Ensure input is (B, C, H, W)
        if x.dim() == 3:
            x = x.transpose(1, 3)  # (B, H, W, C) -> (B, C, H, W)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.relu(x)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        features = self.forward(observation)
        logits = self.action_head(features)
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self.forward(observation)
        return self.value_head(features).squeeze(-1)


class NatureCNN(BaseAgentModel):
    """
    CNN architecture from the Nature DQN paper.
    
    Reference: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015.
    
    Architecture:
        Conv(32, 8x8, stride=4) -> ReLU
        Conv(64, 4x4, stride=2) -> ReLU
        Conv(64, 3x3, stride=1) -> ReLU
        FC(512) -> ReLU
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_shape: Tuple[int, int, int] = (4, 84, 84),  # Grayscale stacked frames
    ):
        super().__init__(config)
        
        self.image_shape = image_shape
        self.channels, self.height, self.width = image_shape
        
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Compute flat dimension
        self._feature_size = self._get_conv_output_size()
        
        self.fc = nn.Linear(self._feature_size, 512)
        
        # Output heads
        self.action_head = nn.Linear(512, config.action_dim)
        self.value_head = nn.Linear(512, 1)
        
        # Initialize
        orthogonal_init_(self.action_head)
        orthogonal_init_(self.value_head)
    
    def _get_conv_output_size(self) -> int:
        """Calculate output size of conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, *self.image_shape)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through network."""
        if x.dim() == 3:
            x = x.transpose(1, 3)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        features = self.forward(observation)
        q_values = self.action_head(features)
        
        if deterministic:
            return torch.argmax(q_values, dim=-1)
        else:
            return torch.argmax(q_values, dim=-1)  # Epsilon-greedy handled externally
    
    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions."""
        features = self.forward(observation)
        return self.action_head(features)
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate (max Q-value)."""
        q_values = self.get_q_values(observation)
        return torch.max(q_values, dim=-1).values


class ResidualBlock(nn.Module):
    """Residual block for ResNet."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNetNetwork(BaseAgentModel):
    """
    ResNet-based CNN for visual agents.
    
    Uses residual blocks for deeper networks with better gradient flow.
    
    Args:
        config: Model configuration
        image_shape: Input image shape
        num_blocks: Number of residual blocks
        base_channels: Base number of channels
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_shape: Tuple[int, int, int] = (3, 84, 84),
        num_blocks: int = 3,
        base_channels: int = 32,
    ):
        super().__init__(config)
        
        self.image_shape = image_shape
        self.channels, self.height, self.width = image_shape
        
        # Initial conv
        self.conv1 = nn.Conv2d(self.channels, base_channels, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*blocks)
        
        # Compute feature size
        with torch.no_grad():
            x = torch.zeros(1, *image_shape)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.res_blocks(x)
            flat_dim = x.view(1, -1).shape[1]
        
        self.fc = nn.Linear(flat_dim, 512)
        
        # Heads
        self.action_head = nn.Linear(512, config.action_dim)
        self.value_head = nn.Linear(512, 1)
        
        orthogonal_init_(self.action_head)
        orthogonal_init_(self.value_head)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through ResNet."""
        if x.dim() == 3:
            x = x.transpose(1, 3)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        features = self.forward(observation)
        logits = self.action_head(features)
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self.forward(observation)
        return self.value_head(features).squeeze(-1)
