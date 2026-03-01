"""
Recurrent Neural Networks for sequential decision making.

Provides RNN, LSTM, and GRU-based agent architectures
for handling temporal dependencies in observations.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.utils.initialization import orthogonal_init_


class RNNNetwork(BaseAgentModel):
    """
    RNN-based agent for sequential observations.
    
    Args:
        config: Model configuration
        rnn_type: Type of RNN ('rnn', 'lstm', 'gru')
        hidden_size: RNN hidden size
        num_layers: Number of RNN layers
        bidirectional: Whether to use bidirectional RNN
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        rnn_type: str = "lstm",
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__(config)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, hidden_size)
        
        # RNN
        rnn_cls = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[rnn_type.lower()]
        
        self.rnn = rnn_cls(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        # Output layers
        rnn_output_dim = hidden_size * self.num_directions
        self.action_head = nn.Linear(rnn_output_dim, config.action_dim)
        self.value_head = nn.Linear(rnn_output_dim, 1)
        
        orthogonal_init_(self.action_head)
        orthogonal_init_(self.value_head)
        
        # Hidden state storage
        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward through RNN.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            hidden_state: Previous hidden state
        
        Returns:
            Output and new hidden state
        """
        # Ensure 3D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Project input
        x = F.relu(self.input_proj(x))
        
        # RNN forward
        output, hidden = self.rnn(x, hidden_state)
        
        # Take last output
        output = output[:, -1, :]
        
        return output, hidden
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        output, hidden = self.forward(observation, self._hidden_state)
        self._hidden_state = hidden
        
        logits = self.action_head(output)
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        output, _ = self.forward(observation, self._hidden_state)
        return self.value_head(output).squeeze(-1)
    
    def reset(self) -> None:
        """Reset hidden state."""
        self._hidden_state = None
    
    def get_hidden_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get current hidden state."""
        return self._hidden_state
    
    def set_hidden_state(
        self,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Set hidden state."""
        self._hidden_state = state


class LSTMAgent(BaseAgentModel):
    """
    LSTM-based agent with separate actor and critic.
    
    Designed for PPO and other policy gradient methods
    that need both action probabilities and value estimates.
    
    Args:
        config: Model configuration
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        super().__init__(config)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Actor-Critic heads
        self.actor = nn.Linear(hidden_size, config.action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        
        orthogonal_init_(self.actor, gain=1.0)
        orthogonal_init_(self.critic, gain=1.0)
        
        # Hidden state
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            (actor_output, critic_output, new_hidden)
        """
        # Ensure 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take last timestep
        out = lstm_out[:, -1, :]
        
        actor_out = self.actor(out)
        critic_out = self.critic(out).squeeze(-1)
        
        return actor_out, critic_out, hidden
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        actor_out, _, hidden = self.forward(observation, self._hidden)
        self._hidden = hidden
        
        logits = actor_out
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        _, critic_out, _ = self.forward(observation, self._hidden)
        return critic_out
    
    def get_action_and_value(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and value in single forward pass."""
        actor_out, critic_out, hidden = self.forward(observation, self._hidden)
        self._hidden = hidden
        
        logits = actor_out
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        
        return action, critic_out
    
    def get_log_prob(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probability of action."""
        actor_out, _, _ = self.forward(observation, self._hidden)
        logits = actor_out
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action)
    
    def get_entropy(self, observation: torch.Tensor) -> torch.Tensor:
        """Get entropy of action distribution."""
        actor_out, _, _ = self.forward(observation, self._hidden)
        logits = actor_out
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()
    
    def reset(self) -> None:
        """Reset hidden state."""
        device = next(self.parameters()).device
        self._hidden = (
            torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
        )
    
    def get_hidden_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get hidden state."""
        return self._hidden
    
    def set_hidden_state(
        self,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Set hidden state."""
        self._hidden = state


class GRUAgent(BaseAgentModel):
    """
    GRU-based agent for sequential decision making.
    
    Simpler alternative to LSTM with fewer parameters.
    
    Args:
        config: Model configuration
        hidden_size: GRU hidden size
        num_layers: Number of GRU layers
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        super().__init__(config)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Output heads
        self.action_head = nn.Linear(hidden_size, config.action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        
        orthogonal_init_(self.action_head)
        orthogonal_init_(self.value_head)
        
        # Hidden state
        self._hidden: Optional[torch.Tensor] = None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        gru_out, hidden = self.gru(x, hidden)
        out = gru_out[:, -1, :]
        
        return out, hidden
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        out, hidden = self.forward(observation, self._hidden)
        self._hidden = hidden
        
        logits = self.action_head(out)
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        out, _ = self.forward(observation, self._hidden)
        return self.value_head(out).squeeze(-1)
    
    def reset(self) -> None:
        """Reset hidden state."""
        device = next(self.parameters()).device
        self._hidden = torch.zeros(
            self.num_layers, 1, self.hidden_size, device=device
        )
    
    def get_hidden_state(self) -> Optional[torch.Tensor]:
        """Get hidden state."""
        return self._hidden
    
    def set_hidden_state(self, state: Optional[torch.Tensor]) -> None:
        """Set hidden state."""
        self._hidden = state
