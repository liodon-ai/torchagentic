"""
Neural Turing Machine (NTM) implementation.

Reference: Graves et al., "Neural Turing Machines", arXiv 2014.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralTuringMachine(nn.Module):
    """
    Neural Turing Machine with differentiable memory.
    
    Features:
    - Content-based addressing
    - Location-based addressing (shifts)
    - Sharpening
    - Read and write heads
    
    Args:
        input_size: Input feature size
        memory_size: Number of memory slots
        memory_dim: Dimension of each memory slot
        num_reads: Number of read heads
        num_writes: Number of write heads
        controller_type: Type of controller ('lstm' or 'gru')
        controller_hidden: Controller hidden size
    """
    
    def __init__(
        self,
        input_size: int,
        memory_size: int = 128,
        memory_dim: int = 64,
        num_reads: int = 4,
        num_writes: int = 1,
        controller_type: str = "lstm",
        controller_hidden: int = 256,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.controller_hidden = controller_hidden
        
        # Controller (LSTM)
        controller_input = input_size + num_reads * memory_dim
        if controller_type == "lstm":
            self.controller = nn.LSTM(controller_input, controller_hidden, batch_first=True)
        else:
            self.controller = nn.GRU(controller_input, controller_hidden, batch_first=True)
        
        self.controller_type = controller_type
        
        # Memory
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        nn.init.xavier_uniform_(self.memory)
        
        # Read heads
        self.read_weights = nn.Parameter(torch.zeros(1, num_reads, memory_size))
        nn.init.constant_(self.read_weights, 1.0 / memory_size)
        
        # Write head parameters
        self.write_strength = nn.Parameter(torch.ones(1))
        self.erase_bias = nn.Parameter(torch.zeros(1))
        
        # Interface vector parameters
        interface_size = (
            memory_dim * num_writes +  # Write values
            memory_dim * num_writes +  # Erase vectors
            1 +  # Write strength
            3 +  # Shift weights (left, stay, right)
            1 +  # Interpolation
            1 +  # Key strength
            memory_dim  # Key
        )
        self.interface_proj = nn.Linear(controller_hidden, interface_size)
        
        # Output projection
        self.output_proj = nn.Linear(controller_hidden + num_reads * memory_dim, controller_hidden)
        
        # Hidden state
        self._hidden: Optional[Tuple] = None
        self._last_weights: Optional[torch.Tensor] = None
    
    def _content_address(
        self,
        key: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        """Content-based addressing."""
        # Cosine similarity
        memory_norm = F.normalize(self.memory.unsqueeze(0), dim=-1)
        key_norm = F.normalize(key.unsqueeze(1), dim=-1)
        
        similarity = torch.matmul(key_norm, memory_norm.transpose(-2, -1))
        weights = F.softmax(similarity * strength.unsqueeze(-1), dim=-1)
        
        return weights
    
    def _convolve(
        self,
        weights: torch.Tensor,
        shift_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Circular convolution for location-based addressing."""
        batch_size, num_heads, mem_size = weights.shape
        
        # Pad for circular convolution
        padded = F.pad(weights, (1, 1), mode="circular")
        
        # Convolution kernels
        kernels = shift_weights.unsqueeze(-1)  # (batch, num_heads, 3, 1)
        
        # Apply convolution
        result = (
            padded[:, :, :-2].unsqueeze(-1) * kernels[:, :, 0:1, :] +
            padded[:, :, 1:-1].unsqueeze(-1) * kernels[:, :, 1:2, :] +
            padded[:, :, 2:].unsqueeze(-1) * kernels[:, :, 2:3, :]
        ).squeeze(-1)
        
        return result
    
    def _interpolate(
        self,
        prev_weights: torch.Tensor,
        new_weights: torch.Tensor,
        interp: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between previous and current weights."""
        return (1 - interp.unsqueeze(-1)) * prev_weights + interp.unsqueeze(-1) * new_weights
    
    def _sharpen(
        self,
        weights: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Sharpen weights."""
        weights = weights ** gamma.unsqueeze(-1)
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            x: Input (batch, seq_len, input_size)
            hidden: Controller hidden state
        
        Returns:
            (output, new_hidden)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state
        if hidden is None:
            hidden = self._hidden
        
        # Initialize weights
        if self._last_weights is None:
            self._last_weights = torch.ones(
                batch_size, self.num_reads, self.memory_size,
                device=x.device,
            ) / self.memory_size
        
        # Read from memory
        read_vectors = self.read(self._last_weights)
        read_flat = read_vectors.reshape(batch_size, -1)
        
        # Controller input
        controller_in = torch.cat([x, read_flat], dim=-1)
        
        if x.dim() == 2:
            controller_in = controller_in.unsqueeze(1)
        
        # Controller forward
        if self.controller_type == "lstm":
            controller_out, hidden = self.controller(controller_in, hidden)
        else:
            controller_out, hidden = self.controller(controller_in)
        
        controller_out = controller_out[:, -1, :]  # Last timestep
        
        # Get interface vector
        interface = self.interface_proj(controller_out)
        
        # Parse interface vector
        offset = 0
        
        # Write values
        write_values = interface[:, offset:offset + self.memory_dim * self.num_writes]
        write_values = write_values.reshape(batch_size, self.num_writes, self.memory_dim)
        offset += self.memory_dim * self.num_writes
        
        # Erase vectors
        erase_vectors = interface[:, offset:offset + self.memory_dim * self.num_writes]
        erase_vectors = torch.sigmoid(erase_vectors.reshape(batch_size, self.num_writes, self.memory_dim))
        offset += self.memory_dim * self.num_writes
        
        # Write strength
        write_strength = F.softplus(interface[:, offset:offset + 1])
        offset += 1
        
        # Shift weights
        shift_weights = F.softmax(interface[:, offset:offset + 3].reshape(batch_size, 1, 3), dim=-1)
        offset += 3
        
        # Interpolation
        interp = torch.sigmoid(interface[:, offset:offset + 1])
        offset += 1
        
        # Key strength
        key_strength = F.softplus(interface[:, offset:offset + 1])
        offset += 1
        
        # Key
        key = interface[:, offset:offset + self.memory_dim]
        
        # Content-based addressing
        content_weights = self._content_address(key, key_strength)
        
        # Location-based addressing (for each read head)
        new_weights = []
        for i in range(self.num_reads):
            prev_w = self._last_weights[:, i:i+1, :]
            
            # Interpolate
            interp_w = self._interpolate(prev_w, content_weights[:, i:i+1, :], interp[:, :1])
            
            # Convolve
            conv_w = self._convolve(interp_w, shift_weights)
            
            # Sharpen
            sharp_w = self._sharpen(conv_w, key_strength[:, :1])
            
            new_weights.append(sharp_w)
        
        new_weights = torch.cat(new_weights, dim=1)
        self._last_weights = new_weights
        
        # Write to memory
        self._write_to_memory(write_values, erase_vectors, new_weights[:, :self.num_writes, :], write_strength)
        
        # Read
        read_vectors = self.read(new_weights)
        read_flat = read_vectors.reshape(batch_size, -1)
        
        # Output
        output = torch.cat([controller_out, read_flat], dim=-1)
        output = self.output_proj(output)
        
        self._hidden = hidden
        
        return output, hidden
    
    def _write_to_memory(
        self,
        values: torch.Tensor,
        erase: torch.Tensor,
        weights: torch.Tensor,
        strength: torch.Tensor,
    ) -> None:
        """Write to memory with erase and add."""
        # Erase
        erase_matrix = torch.matmul(
            weights.transpose(1, 2),
            erase * strength.unsqueeze(-1),
        ).mean(0)
        self.memory.data *= (1 - erase_matrix)
        
        # Add
        add_matrix = torch.matmul(
            weights.transpose(1, 2),
            values * strength.unsqueeze(-1),
        ).mean(0)
        self.memory.data += add_matrix
    
    def read(self, weights: torch.Tensor) -> torch.Tensor:
        """Read from memory."""
        return torch.matmul(weights, self.memory.unsqueeze(0).expand(weights.size(0), -1, -1))
    
    def reset(self) -> None:
        """Reset memory and hidden state."""
        nn.init.xavier_uniform_(self.memory)
        nn.init.constant_(self.read_weights, 1.0 / self.memory_size)
        self._hidden = None
        self._last_weights = None
    
    def get_hidden_state(self) -> Optional[Tuple]:
        """Get hidden state."""
        return self._hidden
    
    def set_hidden_state(self, state: Optional[Tuple]) -> None:
        """Set hidden state."""
        self._hidden = state
