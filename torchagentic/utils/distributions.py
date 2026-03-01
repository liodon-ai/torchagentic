"""
Probability distributions for RL agents.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class TanhNormal(dist.Normal):
    """
    Normal distribution with tanh transformation.
    
    Used for continuous action spaces with bounded actions.
    
    Args:
        loc: Mean
        scale: Standard deviation
    """
    
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        super().__init__(loc, scale)
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample and apply tanh."""
        x = super().sample(sample_shape)
        return torch.tanh(x)
    
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Reparameterized sample with tanh."""
        x = super().rsample(sample_shape)
        return torch.tanh(x)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Log probability with change of variables.
        
        Args:
            value: Value in (-1, 1) after tanh
        """
        # Inverse tanh
        pre_tanh = torch.atanh(value.clamp(-0.9999, 0.9999))
        
        # Normal log prob
        normal_log_prob = super().log_prob(pre_tanh)
        
        # Jacobian correction
        log_jacobian = torch.log(1 - value ** 2 + 1e-6)
        
        return normal_log_prob - log_jacobian
    
    def entropy(self) -> torch.Tensor:
        """Entropy with tanh correction."""
        # Sample and compute
        x = self.rsample()
        log_prob = self.log_prob(x)
        return -log_prob


class DiagGaussian:
    """
    Diagonal Gaussian distribution.
    
    Wrapper for creating Gaussian distributions with
    proper handling of log_std.
    
    Args:
        loc: Mean tensor
        scale_tril: Lower triangular Cholesky factor (optional)
    """
    
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ):
        self.loc = loc
        self.scale = scale
        self.dist = dist.Normal(loc, scale)
    
    def sample(self) -> torch.Tensor:
        """Sample from distribution."""
        return self.dist.sample()
    
    def rsample(self) -> torch.Tensor:
        """Reparameterized sample."""
        return self.dist.rsample()
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        return self.dist.log_prob(value).sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """Entropy."""
        return self.dist.entropy().sum(dim=-1)
    
    def kl(self, other: "DiagGaussian") -> torch.Tensor:
        """
        KL divergence with another Gaussian.
        
        Args:
            other: Other distribution
        """
        return dist.kl_divergence(self.dist, other.dist).sum(dim=-1)


class Categorical(dist.Categorical):
    """
    Categorical distribution with logits.
    
    Extended with additional utilities.
    """
    
    def __init__(self, logits: torch.Tensor = None, probs: torch.Tensor = None):
        super().__init__(logits=logits, probs=probs)
    
    def mode(self) -> torch.Tensor:
        """Get mode (most likely value)."""
        return torch.argmax(self.logits, dim=-1)
    
    def log_prob_of_mode(self) -> torch.Tensor:
        """Get log probability of mode."""
        mode = self.mode()
        return self.log_prob(mode)


class MultiCategorical:
    """
    Multiple independent categorical distributions.
    
    Used for multi-discrete action spaces.
    
    Args:
        logits: Logits tensor (batch, num_categories, max_options)
    """
    
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.dists = [
            dist.Categorical(logits=logits[:, i, :])
            for i in range(logits.shape[1])
        ]
    
    def sample(self) -> torch.Tensor:
        """Sample from all distributions."""
        return torch.stack([d.sample() for d in self.dists], dim=-1)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        return sum(d.log_prob(value[:, i]) for i, d in enumerate(self.dists))
    
    def entropy(self) -> torch.Tensor:
        """Entropy."""
        return sum(d.entropy() for d in self.dists)
    
    def mode(self) -> torch.Tensor:
        """Get mode for each distribution."""
        return torch.stack([d.mode() for d in self.dists], dim=-1)


class Beta(dist.Beta):
    """
    Beta distribution for continuous actions.
    
    Alternative to Gaussian for bounded continuous actions.
    
    Args:
        alpha: Alpha parameter
        beta: Beta parameter
    """
    
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__(alpha, beta)
    
    @classmethod
    def from_mean_std(
        cls,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> "Beta":
        """
        Create Beta from mean and std.
        
        Args:
            mean: Mean in (0, 1)
            std: Standard deviation
        """
        # Convert mean/std to alpha/beta
        var = std ** 2
        mean_sq = mean ** 2
        
        alpha = ((1 - mean) / var - 1 / mean) * mean_sq
        beta = alpha * (1 / mean - 1)
        
        return cls(alpha.clamp(1.0, 100.0), beta.clamp(1.0, 100.0))
