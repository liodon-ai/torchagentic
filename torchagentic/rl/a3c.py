"""
A3C (Asynchronous Advantage Actor-Critic) models.

Implements network architectures for A3C training.
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.utils.initialization import orthogonal_init_


class A3CNetwork(BaseAgentModel):
    """
    A3C network with shared trunk and separate actor/critic heads.
    
    Reference: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", ICML 2016.
    
    Args:
        config: Model configuration
        image_input: Whether input is image
        use_lstm: Whether to add LSTM layer
        lstm_hidden: LSTM hidden size
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_input: bool = False,
        use_lstm: bool = False,
        lstm_hidden: int = 256,
    ):
        super().__init__(config)
        
        self.image_input = image_input
        self.use_lstm = use_lstm
        
        if image_input:
            # CNN trunk (Nature-style)
            self.conv1 = nn.Conv2d(config.input_dim, 16, 8, stride=4)
            self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
            
            # Calculate flat dim
            with torch.no_grad():
                dummy = torch.zeros(1, config.input_dim, 84, 84)
                x = F.relu(self.conv1(dummy))
                x = F.relu(self.conv2(x))
                flat_dim = x.view(1, -1).shape[1]
        else:
            # MLP trunk
            self.fc1 = nn.Linear(config.input_dim, config.hidden_dims[0])
            flat_dim = config.hidden_dims[0]
        
        # LSTM (optional)
        if use_lstm:
            self.lstm = nn.LSTM(flat_dim, lstm_hidden, batch_first=True)
            trunk_output_dim = lstm_hidden
        else:
            self.lstm = None
            trunk_output_dim = flat_dim
        
        # Actor head
        self.actor = nn.Linear(trunk_output_dim, config.action_dim)
        orthogonal_init_(self.actor, gain=0.01)
        
        # Critic head
        self.critic = nn.Linear(trunk_output_dim, 1)
        orthogonal_init_(self.critic, gain=1.0)
        
        # Hidden state for LSTM
        self._hidden: Optional[tuple] = None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            hidden: LSTM hidden state (if using LSTM)
        
        Returns:
            (actor_logits, value, new_hidden)
        """
        if self.image_input:
            if x.dim() == 3:
                x = x.transpose(1, 3)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
        else:
            x = F.relu(self.fc1(x))
        
        # LSTM
        if self.use_lstm:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x, hidden = self.lstm(x, hidden)
            x = x[:, -1, :]
        
        # Heads
        actor_logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        
        return actor_logits, value, hidden
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        logits, _, hidden = self.forward(observation, self._hidden)
        self._hidden = hidden
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            d = dist.Categorical(logits=logits)
            return d.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        _, value, _ = self.forward(observation, self._hidden)
        return value
    
    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, and value.
        
        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value, hidden = self.forward(observation, self._hidden)
        self._hidden = hidden
        
        d = dist.Categorical(logits=logits)
        
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = d.sample()
        
        log_prob = d.log_prob(action)
        entropy = d.entropy()
        
        return action, log_prob, entropy, value
    
    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions.
        
        Returns:
            (log_prob, entropy, value)
        """
        logits, value, _ = self.forward(observation, self._hidden)
        d = dist.Categorical(logits=logits)
        
        log_prob = d.log_prob(action)
        entropy = d.entropy()
        
        return log_prob, entropy, value
    
    def reset(self) -> None:
        """Reset LSTM hidden state."""
        if self.use_lstm:
            device = next(self.parameters()).device
            self._hidden = (
                torch.zeros(1, 1, self.lstm.hidden_size, device=device),
                torch.zeros(1, 1, self.lstm.hidden_size, device=device),
            )
    
    def get_hidden_state(self) -> Optional[tuple]:
        """Get LSTM hidden state."""
        return self._hidden
    
    def set_hidden_state(self, state: Optional[tuple]) -> None:
        """Set LSTM hidden state."""
        self._hidden = state


class A3CLSTM(BaseAgentModel):
    """
    A3C with LSTM for memory-based decision making.
    
    Specifically designed for partially observable environments.
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        image_input: bool = False,
        lstm_hidden: int = 512,
        num_layers: int = 1,
    ):
        super().__init__(config)
        
        self.image_input = image_input
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        
        if image_input:
            self.conv1 = nn.Conv2d(config.input_dim, 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
            
            with torch.no_grad():
                dummy = torch.zeros(1, config.input_dim, 84, 84)
                x = F.relu(self.conv1(dummy))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                conv_out_dim = x.view(1, -1).shape[1]
            
            self.fc_input = nn.Linear(conv_out_dim, lstm_hidden)
        else:
            self.fc_input = nn.Linear(config.input_dim, lstm_hidden)
        
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=num_layers)
        
        self.actor = nn.Linear(lstm_hidden, config.action_dim)
        self.critic = nn.Linear(lstm_hidden, 1)
        
        orthogonal_init_(self.actor, gain=0.01)
        orthogonal_init_(self.critic, gain=1.0)
        
        self._hidden: Optional[tuple] = None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Forward pass."""
        if self.image_input:
            if x.dim() == 3:
                x = x.transpose(1, 3)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc_input(x))
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x, hidden = self.lstm(x, hidden)
        x = x[-1]
        
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        
        return logits, value, hidden
    
    def get_action(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action."""
        logits, _, hidden = self.forward(observation, self._hidden)
        self._hidden = hidden
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            d = dist.Categorical(logits=logits)
            return d.sample()
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value."""
        _, value, _ = self.forward(observation, self._hidden)
        return value
    
    def reset(self) -> None:
        """Reset hidden state."""
        device = next(self.parameters()).device
        self._hidden = (
            torch.zeros(self.num_layers, 1, self.lstm_hidden, device=device),
            torch.zeros(self.num_layers, 1, self.lstm_hidden, device=device),
        )
    
    def get_hidden_state(self) -> Optional[tuple]:
        """Get hidden state."""
        return self._hidden
    
    def set_hidden_state(self, state: Optional[tuple]) -> None:
        """Set hidden state."""
        self._hidden = state
