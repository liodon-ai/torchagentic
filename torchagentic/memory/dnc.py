"""
Differentiable Neural Computer (DNC) implementation.

Reference: Graves et al., "Hybrid computing using a neural network with dynamic external memory", Nature 2016.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNC(nn.Module):
    """
    Differentiable Neural Computer.
    
    An advanced memory-augmented network with:
    - Dynamic memory allocation
    - Temporal link matrix
    - Multiple read/write heads
    - Usage tracking
    
    Args:
        input_size: Input feature size
        memory_size: Number of memory slots
        memory_dim: Dimension of each memory slot
        num_reads: Number of read heads
        num_writes: Number of write heads
        controller_hidden: Controller LSTM hidden size
    """
    
    def __init__(
        self,
        input_size: int,
        memory_size: int = 256,
        memory_dim: int = 64,
        num_reads: int = 4,
        num_writes: int = 1,
        controller_hidden: int = 256,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.controller_hidden = controller_hidden
        
        # Controller
        controller_input = input_size + num_reads * memory_dim
        self.controller = nn.LSTM(controller_input, controller_hidden, batch_first=True)
        
        # Memory
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        nn.init.xavier_uniform_(self.memory)
        
        # Usage tracking
        self.usage = nn.Parameter(torch.zeros(memory_size))
        
        # Temporal link matrix
        self.link_matrix = nn.Parameter(torch.zeros(memory_size, memory_size))
        self.forward_weights = nn.Parameter(torch.zeros(memory_size, memory_size))
        self.backward_weights = nn.Parameter(torch.zeros(memory_size, memory_size))
        
        # Read weights
        self.read_weights = nn.Parameter(torch.zeros(1, num_reads, memory_size))
        nn.init.constant_(self.read_weights, 1.0 / memory_size)
        
        # Write weights
        self.write_weights = nn.Parameter(torch.zeros(1, num_writes, memory_size))
        nn.init.constant_(self.write_weights, 1.0 / memory_size)
        
        # Allocation gate
        self.allocation_gate = nn.Linear(controller_hidden, 1)
        
        # Interface vector projection
        interface_size = self._get_interface_size()
        self.interface_proj = nn.Linear(controller_hidden, interface_size)
        
        # Output
        self.output_proj = nn.Linear(controller_hidden + num_reads * memory_dim, controller_hidden)
        
        # State
        self._hidden: Optional[Tuple] = None
        self._prev_read_weights: Optional[torch.Tensor] = None
        self._prev_write_weights: Optional[torch.Tensor] = None
        self._prev_link_matrix: Optional[torch.Tensor] = None
    
    def _get_interface_size(self) -> int:
        """Calculate interface vector size."""
        return (
            self.memory_dim * self.num_writes +  # Write values
            self.memory_dim * self.num_writes +  # Erase vectors
            self.num_writes +  # Write gates
            1 +  # Allocation gate
            self.num_writes +  # Free gates
            3 * self.num_reads +  # Read modes (content, backward, forward)
            self.num_reads +  # Read strengths
            self.memory_dim * self.num_reads  # Read keys
        )
    
    def _content_address(
        self,
        key: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        """Content-based addressing."""
        memory_norm = F.normalize(self.memory.unsqueeze(0), dim=-1)
        key_norm = F.normalize(key.unsqueeze(1), dim=-1)
        
        similarity = torch.matmul(key_norm, memory_norm.transpose(-2, -1))
        weights = F.softmax(similarity * strength.unsqueeze(-1), dim=-1)
        
        return weights
    
    def _get_allocation_weights(self, gate: torch.Tensor) -> torch.Tensor:
        """Get memory allocation weights."""
        # Sort by usage
        usage = self.usage.unsqueeze(0)
        _, indices = torch.sort(usage, dim=-1, descending=False)
        
        # Create allocation weights (prefer less used locations)
        sorted_usage = torch.gather(usage, -1, indices)
        cumprod = torch.cumprod(1 - sorted_usage, dim=-1)
        
        allocation = torch.zeros_like(sorted_usage)
        allocation[..., 0] = sorted_usage[..., 0]
        allocation[..., 1:] = sorted_usage[..., 1:] * cumprod[..., :-1]
        
        # Unsort
        unsort_indices = torch.argsort(indices, dim=-1)
        allocation = torch.gather(allocation, -1, unsort_indices)
        
        return allocation * gate.unsqueeze(-1)
    
    def _update_links(
        self,
        write_weights: torch.Tensor,
    ) -> None:
        """Update temporal link matrix."""
        batch_size = write_weights.shape[0]
        
        # Outer product for links
        link_update = torch.matmul(
            write_weights.transpose(1, 2),  # (batch, num_writes, mem_size)
            write_weights,  # (batch, num_writes, mem_size)
        )  # (batch, mem_size, mem_size)
        
        # Remove self-links
        mask = 1 - torch.eye(self.memory_size, device=write_weights.device)
        link_update = link_update * mask.unsqueeze(0)
        
        # Update link matrix (with decay)
        decay = 0.9
        self.link_matrix.data = self.link_matrix.data * decay + link_update.mean(0) * (1 - decay)
    
    def _get_temporal_weights(
        self,
        prev_read_weights: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        """Get temporal weights (forward or backward)."""
        if mode == 0:  # Content-based
            return None
        
        # Use link matrix
        if mode == 1:  # Backward
            weights = torch.matmul(prev_read_weights, self.link_matrix)
        else:  # Forward
            weights = torch.matmul(prev_read_weights, self.link_matrix.transpose(-2, -1))
        
        return weights
    
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
        
        if hidden is None:
            hidden = self._hidden
        
        # Initialize state
        if self._prev_read_weights is None:
            self._prev_read_weights = torch.ones(
                batch_size, self.num_reads, self.memory_size,
                device=x.device,
            ) / self.memory_size
        
        if self._prev_write_weights is None:
            self._prev_write_weights = torch.ones(
                batch_size, self.num_writes, self.memory_size,
                device=x.device,
            ) / self.memory_size
        
        # Read from memory
        read_vectors = self.read(self._prev_read_weights)
        read_flat = read_vectors.reshape(batch_size, -1)
        
        # Controller input
        controller_in = torch.cat([x, read_flat], dim=-1)
        
        if x.dim() == 2:
            controller_in = controller_in.unsqueeze(1)
        
        # Controller
        controller_out, hidden = self.controller(controller_in, hidden)
        controller_out = controller_out[:, -1, :]
        
        # Interface vector
        interface = self.interface_proj(controller_out)
        
        # Parse interface
        offset = 0
        
        write_values = interface[:, offset:offset + self.memory_dim * self.num_writes]
        write_values = write_values.reshape(batch_size, self.num_writes, self.memory_dim)
        offset += self.memory_dim * self.num_writes
        
        erase_vectors = interface[:, offset:offset + self.memory_dim * self.num_writes]
        erase_vectors = torch.sigmoid(erase_vectors.reshape(batch_size, self.num_writes, self.memory_dim))
        offset += self.memory_dim * self.num_writes
        
        write_gates = torch.sigmoid(interface[:, offset:offset + self.num_writes])
        offset += self.num_writes
        
        allocation_gate = torch.sigmoid(interface[:, offset:offset + 1])
        offset += 1
        
        free_gates = torch.sigmoid(interface[:, offset:offset + self.num_writes])
        offset += self.num_writes
        
        read_modes = interface[:, offset:offset + 3 * self.num_reads]
        read_modes = read_modes.reshape(batch_size, self.num_reads, 3)
        read_modes = F.softmax(read_modes, dim=-1)
        offset += 3 * self.num_reads
        
        read_strengths = F.softplus(interface[:, offset:offset + self.num_reads])
        offset += self.num_reads
        
        read_keys = interface[:, offset:offset + self.memory_dim * self.num_reads]
        read_keys = read_keys.reshape(batch_size, self.num_reads, self.memory_dim)
        
        # Get write weights
        content_weights = self._content_address(
            write_values.mean(1),
            torch.ones(batch_size, 1, device=x.device),
        )
        allocation_weights = self._get_allocation_weights(allocation_gate.squeeze(-1))
        
        write_weights = (
            write_gates.unsqueeze(-1) * 
            (0.5 * content_weights + 0.5 * allocation_weights).unsqueeze(1)
        )
        
        # Update links
        self._update_links(write_weights)
        
        # Write to memory
        self._write(write_values, erase_vectors, write_weights, write_gates)
        
        # Get read weights
        new_read_weights = []
        for i in range(self.num_reads):
            # Content-based
            content_w = self._content_address(read_keys[:, i:i+1, :], read_strengths[:, i:i+1])
            
            # Temporal
            backward_w = self._get_temporal_weights(self._prev_read_weights[:, i:i+1, :], 1)
            forward_w = self._get_temporal_weights(self._prev_read_weights[:, i:i+1, :], 2)
            
            # Combine modes
            mode = read_modes[:, i, :]
            combined = (
                mode[:, 0:1] * content_w +
                mode[:, 1:2] * (backward_w if backward_w is not None else content_w) +
                mode[:, 2:3] * (forward_w if forward_w is not None else content_w)
            )
            
            new_read_weights.append(combined)
        
        new_read_weights = torch.cat(new_read_weights, dim=1)
        self._prev_read_weights = new_read_weights
        self._prev_write_weights = write_weights
        
        # Read
        read_vectors = self.read(new_read_weights)
        read_flat = read_vectors.reshape(batch_size, -1)
        
        # Output
        output = torch.cat([controller_out, read_flat], dim=-1)
        output = self.output_proj(output)
        
        self._hidden = hidden
        
        return output, hidden
    
    def _write(
        self,
        values: torch.Tensor,
        erase: torch.Tensor,
        weights: torch.Tensor,
        gates: torch.Tensor,
    ) -> None:
        """Write to memory."""
        # Update usage
        usage_update = weights.mean(1).mean(0)
        self.usage.data = self.usage.data * 0.99 + usage_update * 0.01
        
        # Erase
        erase_matrix = torch.matmul(
            weights.transpose(1, 2),
            erase * gates.unsqueeze(-1),
        ).mean(0)
        self.memory.data *= (1 - erase_matrix)
        
        # Add
        add_matrix = torch.matmul(
            weights.transpose(1, 2),
            values * gates.unsqueeze(-1),
        ).mean(0)
        self.memory.data += add_matrix
    
    def read(self, weights: torch.Tensor) -> torch.Tensor:
        """Read from memory."""
        return torch.matmul(weights, self.memory.unsqueeze(0).expand(weights.size(0), -1, -1))
    
    def reset(self) -> None:
        """Reset all state."""
        nn.init.xavier_uniform_(self.memory)
        self.usage.data.zero_()
        self.link_matrix.data.zero_()
        nn.init.constant_(self.read_weights, 1.0 / self.memory_size)
        nn.init.constant_(self.write_weights, 1.0 / self.memory_size)
        self._hidden = None
        self._prev_read_weights = None
        self._prev_write_weights = None
    
    def get_hidden_state(self) -> Optional[Tuple]:
        """Get hidden state."""
        return self._hidden
    
    def set_hidden_state(self, state: Optional[Tuple]) -> None:
        """Set hidden state."""
        self._hidden = state


# Alias
DifferentiableNeuralComputer = DNC
