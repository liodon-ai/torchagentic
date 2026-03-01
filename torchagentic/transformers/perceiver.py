"""
Perceiver-based agent models.

Implements Perceiver IO architecture for handling diverse input modalities.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.transformers.attention import MultiHeadAttention, TransformerBlock


class PerceiverResampler(nn.Module):
    """
    Perceiver resampler module.
    
    Compresses large inputs into a fixed-size latent representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_latents = num_latents
        
        # Learnable latents
        self.latents = nn.Parameter(torch.zeros(1, num_latents, embed_dim))
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Cross-attention layers
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Self-attention layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.latents, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional input mask
        
        Returns:
            Latent representation (batch, num_latents, embed_dim)
        """
        B = x.shape[0]
        
        # Project input
        x = self.input_proj(x)
        
        # Expand latents
        latents = self.latents.expand(B, -1, -1)
        
        # Cross-attention: latents attend to input
        latents = self.cross_attn(latents, x, x, mask)
        
        # Self-attention on latents
        for block in self.transformer_blocks:
            latents = block(latents)
        
        return self.ln(latents)


class PerceiverAgent(BaseAgentModel):
    """
    Perceiver-based agent for handling large or multimodal inputs.
    
    Uses a bottleneck of latent variables to process inputs efficiently.
    
    Args:
        config: Model configuration
        embed_dim: Embedding dimension
        num_latents: Number of latent variables
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        embed_dim: int = 256,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(config)
        
        self.embed_dim = embed_dim
        
        # Perceiver encoder
        self.encoder = PerceiverResampler(
            input_dim=config.input_dim,
            embed_dim=embed_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, config.action_dim),
        )
        
        # Value decoder
        self.value_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        
        Returns:
            Latent representation (batch, num_latents, embed_dim)
        """
        return self.encoder(x)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        # Ensure 3D
        if observation.dim() == 2:
            observation = observation.unsqueeze(1)
        
        # Encode
        latents = self.forward(observation)
        
        # Pool latents
        pooled = latents.mean(dim=1)
        
        # Decode action
        logits = self.action_decoder(pooled)
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        if observation.dim() == 2:
            observation = observation.unsqueeze(1)
        
        latents = self.forward(observation)
        pooled = latents.mean(dim=1)
        
        return self.value_decoder(pooled).squeeze(-1)
