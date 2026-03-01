"""
QMIX and VDN for cooperative multi-agent RL.

Reference: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", ICML 2018.
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.multiagent.base import MultiAgentBase
from torchagentic.utils.initialization import orthogonal_init_


class QMIXNetwork(MultiAgentBase):
    """
    QMIX network for cooperative multi-agent RL.
    
    Mixes individual agent Q-values using a monotonic mixing network.
    
    Args:
        num_agents: Number of agents
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        hidden_dim: Hidden dimension for agent networks
        mixing_embed_dim: Embedding dimension for mixing network
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        mixing_embed_dim: int = 32,
    ):
        super().__init__(num_agents, obs_dim, action_dim, shared_params=True)
        
        self.hidden_dim = hidden_dim
        self.mixing_embed_dim = mixing_embed_dim
        
        # Agent networks (shared)
        self.agent_network = self._make_agent_network()
        
        # Mixing network
        self.hyper_w1 = nn.Linear(mixing_embed_dim, hidden_dim * num_agents)
        self.hyper_w2 = nn.Linear(mixing_embed_dim, hidden_dim)
        self.hyper_b1 = nn.Linear(mixing_embed_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(mixing_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Final layer
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        orthogonal_init_(self.final_layer)
    
    def _make_agent_network(self) -> nn.Module:
        """Create individual agent network."""
        return nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            observations: (batch, num_agents, obs_dim)
            actions: (batch, num_agents, action_dim) - one-hot actions
            global_state: (batch, state_dim) - global state for mixing
        
        Returns:
            Mixed Q-value (batch,)
        """
        batch_size = observations.shape[0]
        
        # Get individual Q-values
        q_values = []
        for i in range(self.num_agents):
            obs = observations[:, i, :]
            action = actions[:, i, :]  # One-hot
            
            q = self.agent_network(obs)  # (batch, action_dim)
            q_a = (q * action).sum(dim=-1, keepdim=True)  # (batch, 1)
            q_values.append(q_a)
        
        q_values = torch.cat(q_values, dim=-1)  # (batch, num_agents)
        
        # Mixing
        mixed = self._mix(q_values, global_state)
        
        return mixed.squeeze(-1)
    
    def _mix(
        self,
        q_values: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix Q-values monotonically.
        
        Args:
            q_values: (batch, num_agents)
            global_state: (batch, state_dim)
        
        Returns:
            Mixed Q-value
        """
        # Generate mixing weights
        w1 = torch.abs(self.hyper_w1(global_state))  # (batch, hidden * num_agents)
        b1 = self.hyper_b1(global_state)  # (batch, hidden)
        
        w1 = w1.view(-1, self.num_agents, self.hidden_dim).transpose(1, 2)  # (batch, hidden, num_agents)
        
        # First mixing layer
        hidden = F.relu(torch.matmul(q_values.unsqueeze(1), w1) + b1.unsqueeze(-1))
        hidden = hidden.squeeze(1)  # (batch, hidden)
        
        # Second mixing layer
        w2 = torch.abs(self.hyper_w2(global_state)).unsqueeze(-1)  # (batch, hidden, 1)
        b2 = self.hyper_b2(global_state)  # (batch, 1)
        
        mixed = torch.matmul(hidden.unsqueeze(1), w2).squeeze(-1) + b2  # (batch, 1)
        
        return mixed
    
    def get_actions(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get actions for all agents."""
        batch_size = observations.shape[0]
        actions = []
        
        for i in range(self.num_agents):
            obs = observations[:, i, :]
            q = self.agent_network(obs)
            
            if deterministic:
                action = torch.argmax(q, dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=q)
                action = dist.sample()
            
            actions.append(action)
        
        return torch.stack(actions, dim=1)
    
    def get_individual_q_values(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Get individual Q-values for all agents."""
        batch_size = observations.shape[0]
        q_values = []
        
        for i in range(self.num_agents):
            obs = observations[:, i, :]
            q = self.agent_network(obs)
            q_values.append(q)
        
        return torch.stack(q_values, dim=1)  # (batch, num_agents, action_dim)


class VDNNetwork(MultiAgentBase):
    """
    VDN (Value Decomposition Networks).
    
    Simple additive value decomposition: Q_tot = sum(Q_i)
    
    Reference: Sunehag et al., "Value-Decomposition Networks For Cooperative Multi-Agent Learning", AAMAS 2018.
    
    Args:
        num_agents: Number of agents
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        hidden_dim: Hidden dimension
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__(num_agents, obs_dim, action_dim, shared_params=True)
        
        self.hidden_dim = hidden_dim
        
        # Agent networks (shared)
        self.agent_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            observations: (batch, num_agents, obs_dim)
            actions: (batch, num_agents, action_dim) - one-hot
        
        Returns:
            Total Q-value (batch,)
        """
        q_values = self.get_individual_q_values(observations)
        
        # Select Q-values for taken actions
        q_selected = (q_values * actions).sum(dim=-1)  # (batch, num_agents)
        
        # Sum for total Q
        q_total = q_selected.sum(dim=-1)  # (batch,)
        
        return q_total
    
    def get_actions(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get actions for all agents."""
        q_values = self.get_individual_q_values(observations)
        
        if deterministic:
            actions = torch.argmax(q_values, dim=-1)
        else:
            actions = []
            for i in range(self.num_agents):
                dist = torch.distributions.Categorical(logits=q_values[:, i, :])
                actions.append(dist.sample())
            actions = torch.stack(actions, dim=1)
        
        return actions
    
    def get_individual_q_values(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Get individual Q-values for all agents."""
        batch_size = observations.shape[0]
        q_values = []
        
        for i in range(self.num_agents):
            obs = observations[:, i, :]
            q = self.agent_network(obs)
            q_values.append(q)
        
        return torch.stack(q_values, dim=1)  # (batch, num_agents, action_dim)
