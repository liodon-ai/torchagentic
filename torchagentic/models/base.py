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


# Import compile support
try:
    from torchagentic.compile.core import CompileConfig, compile_model, is_compiled
    COMPILE_AVAILABLE = True
except ImportError:
    COMPILE_AVAILABLE = False


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

    def compile(
        self,
        mode: str = "default",
        dynamic: bool = False,
        fullgraph: bool = False,
        backend: str = "inductor",
        warmup: bool = True,
        example_inputs: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> "BaseAgentModel":
        """
        Compile the model with torch.compile() for PyTorch 2.0+.
        
        Args:
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            dynamic: Enable dynamic shapes
            fullgraph: Compile entire graph
            backend: Backend to use
            warmup: Run warmup iteration
            example_inputs: Example inputs for warmup
        
        Returns:
            Self for method chaining
        
        Example:
            >>> model = MLPNetwork(config)
            >>> model.compile(mode="reduce-overhead")
        """
        if not COMPILE_AVAILABLE:
            import warnings
            warnings.warn(
                f"torch.compile() requires PyTorch 2.0+. Current: {torch.__version__}"
            )
            return self
        
        # Create example inputs if not provided
        if example_inputs is None:
            try:
                batch_size = 1 if mode == "reduce-overhead" else 4
                if hasattr(self, "image_input") and self.image_input:
                    shape = (batch_size, *getattr(self, "image_shape", (3, 84, 84)))
                else:
                    shape = (batch_size, self.config.input_dim)
                example_inputs = (torch.randn(shape),)
            except Exception:
                example_inputs = None
        
        # Compile
        compiled = compile_model(
            self,
            config=CompileConfig(
                mode=mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
                backend=backend,
            ),
            example_inputs=example_inputs,
            warmup=warmup,
        )
        
        # Copy compiled state back to self
        self.__dict__.update(compiled.__dict__)
        self._compiled = True
        self._compile_config = CompileConfig(
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
            backend=backend,
        )
        
        return self
    
    @property
    def is_compiled(self) -> bool:
        """Check if model is compiled."""
        return getattr(self, "_compiled", False) or is_compiled(self)
    
    def reset_compile(self) -> None:
        """Reset compilation state."""
        if hasattr(self, "_compiled"):
            self._compiled = False
        if hasattr(self, "_compile_config"):
            self._compile_config = None

    def __repr__(self) -> str:
        compiled_str = " (compiled)" if getattr(self, "_compiled", False) else ""
        return (
            f"{self.__class__.__name__}("
            f"params={self.get_num_params():,}, "
            f"trainable={self.get_trainable_params():,}{compiled_str})"
        )
