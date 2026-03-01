"""
Attention mechanisms for transformer-based agents.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-attention mechanism.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = q @ k.transpose(-2, -1) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with separate Q, K, V projections.
    
    Supports cross-attention between different sequences.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor (batch, seq_len_q, embed_dim)
            key: Key tensor (batch, seq_len_k, embed_dim)
            value: Value tensor (batch, seq_len_k, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch, seq_len_q, embed_dim)
        """
        B, T_q, C = query.shape
        T_k = key.shape[1]
        
        # Project Q, K, V
        q = self.query(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = q @ k.transpose(-2, -1) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T_q, C)
        
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, dropout, bias)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim, bias=bias),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim, bias=bias),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x
