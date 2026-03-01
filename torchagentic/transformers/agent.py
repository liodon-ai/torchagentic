"""
Transformer-based agent models.

Implements transformer architectures for decision making,
including Decision Transformer for offline RL.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.transformers.attention import SelfAttention, TransformerBlock


class TransformerAgent(BaseAgentModel):
    """
    Transformer-based agent for sequential decision making.
    
    Uses self-attention to process sequences of observations.
    
    Args:
        config: Model configuration
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__(config)
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Input embedding
        self.input_embed = nn.Linear(config.input_dim, embed_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        
        # Output heads
        self.action_head = nn.Linear(embed_dim, config.action_dim)
        self.value_head = nn.Linear(embed_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.normal_(self.pos_embed, std=0.02)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        B, T, _ = x.shape
        
        # Embed input
        x = self.input_embed(x)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :T, :]
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        return self.ln(x)
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        # Ensure 3D
        if observation.dim() == 2:
            observation = observation.unsqueeze(1)
        
        features = self.forward(observation)
        
        # Use last token
        features = features[:, -1, :]
        
        logits = self.action_head(features)
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        if observation.dim() == 2:
            observation = observation.unsqueeze(1)
        
        features = self.forward(observation)
        features = features[:, -1, :]
        
        return self.value_head(features).squeeze(-1)


class DecisionTransformer(BaseAgentModel):
    """
    Decision Transformer for offline reinforcement learning.
    
    Reference: Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling", NeurIPS 2021.
    
    Models trajectories as sequences of (state, action, return-to-go).
    
    Args:
        config: Model configuration
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        max_ep_len: Maximum episode length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        embed_dim: int = 128,
        num_heads: int = 1,
        num_layers: int = 3,
        max_seq_len: int = 20,
        max_ep_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__(config)
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.max_ep_len = max_ep_len
        
        # State embedding
        self.state_embed = nn.Linear(config.input_dim, embed_dim)
        
        # Action embedding (for autoregressive generation)
        self.action_embed = nn.Linear(config.action_dim, embed_dim)
        
        # Return-to-go embedding
        self.rtg_embed = nn.Linear(1, embed_dim)
        
        # Positional embeddings
        self.state_pos_embed = nn.Parameter(torch.zeros(1, max_ep_len, embed_dim))
        self.action_pos_embed = nn.Parameter(torch.zeros(1, max_ep_len, embed_dim))
        
        # Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, config.action_dim),
            nn.Tanh(),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.normal_(self.state_pos_embed, std=0.02)
        nn.init.normal_(self.action_pos_embed, std=0.02)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            states: States (batch, seq_len, state_dim)
            actions: Actions (batch, seq_len, action_dim)
            returns_to_go: Returns to go (batch, seq_len, 1)
        
        Returns:
            Predicted actions (batch, seq_len, action_dim)
        """
        B, T, _ = states.shape
        
        # Embed inputs
        state_embeds = self.state_embed(states)
        action_embeds = self.action_embed(actions)
        rtg_embeds = self.rtg_embed(returns_to_go)
        
        # Interleave state, action, rtg embeddings
        # [s1, a1, r1, s2, a2, r2, ...]
        stacked = torch.stack([state_embeds, action_embeds, rtg_embeds], dim=2)
        x = stacked.reshape(B, T * 3, self.embed_dim)
        
        # Add positional embeddings (repeat for each token type)
        pos_embeds = torch.stack([
            self.state_pos_embed[:, :T, :],
            self.action_pos_embed[:, :T, :],
            self.state_pos_embed[:, :T, :],  # Use state pos for rtg
        ], dim=2).reshape(1, T * 3, self.embed_dim)
        
        x = x + pos_embeds
        x = self.dropout(x)
        
        # Causal mask
        mask = torch.tril(torch.ones(T * 3, T * 3, device=x.device)).view(1, 1, T * 3, T * 3)
        
        # Transformer
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.ln(x)
        
        # Extract action predictions (every 3rd token starting from index 1)
        action_preds = self.action_head(x[:, 1::3, :])
        
        return action_preds
    
    def get_action(
        self,
        observation: torch.Tensor,
        returns_to_go: torch.Tensor,
        past_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get action for current timestep.
        
        Args:
            observation: Current observation (batch, state_dim)
            returns_to_go: Current return-to-go (batch, 1)
            past_actions: Past actions in episode (batch, past_len, action_dim)
            deterministic: Whether to use deterministic action
        
        Returns:
            Action (batch, action_dim)
        """
        B = observation.shape[0]
        
        # Prepare sequence
        if past_actions is None:
            # First timestep
            states = observation.unsqueeze(1)
            actions = torch.zeros(B, 1, self.config.action_dim, device=observation.device)
            rtgs = returns_to_go.unsqueeze(1)
        else:
            # Use past trajectory
            T = past_actions.shape[1] + 1
            states = torch.cat([
                torch.zeros(B, T - 1, self.config.input_dim, device=observation.device),
                observation.unsqueeze(1),
            ], dim=1)
            actions = torch.cat([
                past_actions,
                torch.zeros(B, 1, self.config.action_dim, device=observation.device),
            ], dim=1)
            rtgs = returns_to_go.unsqueeze(1).repeat(1, T, 1)
        
        # Get action predictions
        action_preds = self.forward(states, actions, rtgs)
        
        # Get last action prediction
        action = action_preds[:, -1, :]
        
        if deterministic:
            return action
        else:
            # Add small noise for exploration
            noise = torch.randn_like(action) * 0.1
            return action + noise
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Not directly implemented for Decision Transformer."""
        # Could estimate value from returns_to_go
        raise NotImplementedError("Use returns_to_go for value estimation")
