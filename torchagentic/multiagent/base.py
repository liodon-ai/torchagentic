"""
Base class for multi-agent architectures.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import torch
import torch.nn as nn


class MultiAgentBase(nn.Module, ABC):
    """
    Abstract base class for multi-agent architectures.
    
    Provides common functionality for multi-agent systems including:
    - Agent observation/action handling
    - Centralized training with decentralized execution
    - Communication mechanisms
    
    Args:
        num_agents: Number of agents in the system
        obs_dim: Observation dimension per agent
        action_dim: Action dimension per agent
        shared_params: Whether agents share parameters
    """
    
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        shared_params: bool = True,
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.shared_params = shared_params
        
        # Agent states
        self._agent_states: Dict[int, torch.Tensor] = {}
    
    @abstractmethod
    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            observations: Observations for all agents (batch, num_agents, obs_dim)
        
        Returns:
            Outputs for all agents
        """
        pass
    
    @abstractmethod
    def get_actions(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get actions for all agents.
        
        Args:
            observations: Observations for all agents
            deterministic: Whether to use deterministic actions
        
        Returns:
            Actions for all agents (batch, num_agents, action_dim)
        """
        pass
    
    def reset(self) -> None:
        """Reset agent states."""
        self._agent_states = {}
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_agents={self.num_agents}, "
            f"params={self.get_num_params():,})"
        )
