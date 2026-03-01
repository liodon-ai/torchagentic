"""
MADDPG (Multi-Agent DDPG) implementation.

Reference: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS 2017.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.multiagent.base import MultiAgentBase
from torchagentic.utils.initialization import orthogonal_init_


class MADDPGAgent(MultiAgentBase):
    """
    MADDPG agent with centralized critic and decentralized actors.
    
    Each agent has its own actor, but critics have access to
    all agents' observations and actions during training.
    
    Args:
        num_agents: Number of agents
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        hidden_dims: Hidden layer dimensions
        shared_params: Whether agents share parameters
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        shared_params: bool = True,
    ):
        super().__init__(num_agents, obs_dim, action_dim, shared_params)
        
        self.hidden_dims = hidden_dims
        
        # Actor networks (one per agent or shared)
        if shared_params:
            self.actors = nn.ModuleList([self._make_actor()])
        else:
            self.actors = nn.ModuleList([self._make_actor() for _ in range(num_agents)])
        
        # Centralized critic
        # Takes all agents' observations and actions
        total_obs_dim = obs_dim * num_agents
        total_action_dim = action_dim * num_agents
        
        self.critic = self._make_critic(total_obs_dim, total_action_dim)
    
    def _make_actor(self) -> nn.Module:
        """Create actor network."""
        layers = []
        prev_dim = self.obs_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.action_dim))
        layers.append(nn.Tanh())
        
        actor = nn.Sequential(*layers)
        orthogonal_init_(actor[-1], gain=0.01)
        
        return actor
    
    def _make_critic(self, obs_dim: int, action_dim: int) -> nn.Module:
        """Create critic network."""
        layers = []
        prev_dim = obs_dim + action_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        critic = nn.Sequential(*layers)
        orthogonal_init_(critic[-1])
        
        return critic
    
    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through actors.
        
        Args:
            observations: (batch, num_agents, obs_dim)
        
        Returns:
            Actions (batch, num_agents, action_dim)
        """
        batch_size = observations.shape[0]
        actions = []
        
        for i in range(self.num_agents):
            obs = observations[:, i, :]
            actor = self.actors[0] if self.shared_params else self.actors[i]
            action = actor(obs)
            actions.append(action)
        
        return torch.stack(actions, dim=1)
    
    def get_actions(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
        noise: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Get actions for all agents.
        
        Args:
            observations: (batch, num_agents, obs_dim)
            deterministic: Whether to use deterministic actions
            noise: Optional exploration noise std
        
        Returns:
            Actions (batch, num_agents, action_dim)
        """
        actions = self.forward(observations)
        
        if noise is not None and not deterministic:
            actions = actions + torch.randn_like(actions) * noise
            actions = torch.clamp(actions, -1, 1)
        
        return actions
    
    def get_q_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get Q-value from centralized critic.
        
        Args:
            observations: (batch, num_agents, obs_dim)
            actions: (batch, num_agents, action_dim)
        
        Returns:
            Q-value (batch,)
        """
        batch_size = observations.shape[0]
        
        # Flatten all observations and actions
        obs_flat = observations.reshape(batch_size, -1)
        act_flat = actions.reshape(batch_size, -1)
        
        return self.critic(torch.cat([obs_flat, act_flat], dim=-1)).squeeze(-1)
    
    def get_actor(self, agent_id: int) -> nn.Module:
        """Get actor for specific agent."""
        if self.shared_params:
            return self.actors[0]
        return self.actors[agent_id]


class MADDPGCritic(nn.Module):
    """
    Standalone MADDPG critic for ensemble methods.
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        total_obs_dim = obs_dim * num_agents
        total_action_dim = action_dim * num_agents
        
        layers = []
        prev_dim = total_obs_dim + total_action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.critic = nn.Sequential(*layers)
        orthogonal_init_(self.critic[-1])
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size = observations.shape[0]
        
        obs_flat = observations.reshape(batch_size, -1)
        act_flat = actions.reshape(batch_size, -1)
        
        return self.critic(torch.cat([obs_flat, act_flat], dim=-1)).squeeze(-1)
