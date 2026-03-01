"""
Base classes for all agent models in TorchAgentic.

Provides abstract base classes and configuration dataclasses
that all agent architectures should inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """
    Base configuration for agent models.
    
    Attributes:
        input_dim: Dimension of input observations
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        dropout: Dropout probability
        use_layer_norm: Whether to use layer normalization
        use_batch_norm: Whether to use batch normalization
    """
    input_dim: int = 64
    action_dim: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dropout: float = 0.0
    use_layer_norm: bool = False
    use_batch_norm: bool = False
    
    # Optimization settings
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    def get_activation(self) -> nn.Module:
        """Get activation function from name."""
        activations = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        return activations.get(self.activation, nn.ReLU)()


class BaseAgentModel(nn.Module, ABC):
    """
    Abstract base class for all agent models.
    
    All agent architectures should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self._training_steps = 0
        self._total_loss = 0.0
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or
               (batch_size, channels, height, width) for images
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get action from the model.
        
        Args:
            observation: Current observation
            deterministic: Whether to return deterministic action
        
        Returns:
            Action tensor
        """
        pass
    
    @abstractmethod
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for the observation.
        
        Args:
            observation: Current observation
        
        Returns:
            Value estimate
        """
        pass
    
    def reset(self) -> None:
        """Reset internal state (for recurrent models)."""
        pass
    
    def get_hidden_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get hidden state (for recurrent models)."""
        return None
    
    def set_hidden_state(
        self,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Set hidden state (for recurrent models)."""
        pass
    
    def save(self, path: str, include_optimizer: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "training_steps": self._training_steps,
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str, device: Optional[str] = None) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        self._training_steps = checkpoint.get("training_steps", 0)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"params={self.get_num_params():,}, "
            f"trainable={self.get_trainable_params():,})"
        )
