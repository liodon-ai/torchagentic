"""
Normalization layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningNorm(nn.Module):
    """
    Running normalization (like BatchNorm but for RL).
    
    Maintains running mean and variance estimates.
    
    Args:
        num_features: Number of features
        decay: Decay factor for running statistics
        eps: Epsilon for numerical stability
    """
    
    def __init__(
        self,
        num_features: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.decay = decay
        self.eps = eps
        
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("count", torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.training:
            # Update running statistics
            if self.count == 0:
                self.running_mean = x.mean(0).detach()
                self.running_var = x.var(0).detach() + self.eps
                self.count.fill_(1)
            else:
                batch_mean = x.mean(0).detach()
                batch_var = x.var(0).detach() + self.eps
                
                self.running_mean = self.decay * self.running_mean + (1 - self.decay) * batch_mean
                self.running_var = self.decay * self.running_var + (1 - self.decay) * batch_var
                self.count += 1
        
        # Normalize
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
    
    def reset(self) -> None:
        """Reset running statistics."""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.count.zero_()


class LayerNorm2D(nn.Module):
    """
    Layer normalization for 2D inputs (images).
    
    Args:
        num_channels: Number of channels
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        
        self.num_channels = num_channels
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        """
        # Compute mean and variance over C, H, W
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class PixelNorm(nn.Module):
    """
    Pixel-wise normalization (used in StyleGAN).
    
    Normalizes each pixel's feature vector to unit length.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        """
        return x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + self.eps)


class GroupNorm32(nn.GroupNorm):
    """
    GroupNorm with fp32 accumulation for stability.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def get_norm_layer(
    norm_type: str,
    num_features: int,
    **kwargs,
) -> nn.Module:
    """
    Get normalization layer by type.
    
    Args:
        norm_type: Type of normalization
        num_features: Number of features
        **kwargs: Additional arguments
    
    Returns:
        Normalization layer
    """
    norm_types = {
        "batch": lambda: nn.BatchNorm1d(num_features, **kwargs),
        "batch2d": lambda: nn.BatchNorm2d(num_features, **kwargs),
        "layer": lambda: nn.LayerNorm(num_features, **kwargs),
        "layer2d": lambda: LayerNorm2D(num_features, **kwargs),
        "running": lambda: RunningNorm(num_features, **kwargs),
        "pixel": lambda: PixelNorm(**kwargs),
        "none": lambda: nn.Identity(),
    }
    
    if norm_type not in norm_types:
        raise ValueError(f"Unknown norm_type: {norm_type}")
    
    return norm_types[norm_type]()
