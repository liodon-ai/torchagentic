"""
Core memory modules for memory-augmented networks.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryMatrix(nn.Module):
    """
    Differentiable memory matrix.
    
    A learnable memory bank that can be read from and written to
    using attention-based mechanisms.
    
    Args:
        num_slots: Number of memory slots
        slot_size: Size of each memory slot
    """
    
    def __init__(self, num_slots: int = 256, slot_size: int = 128):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_size = slot_size
        
        # Memory matrix
        self.memory = nn.Parameter(torch.zeros(num_slots, slot_size))
        nn.init.xavier_uniform_(self.memory)
    
    def forward(self) -> torch.Tensor:
        """Return memory matrix."""
        return self.memory
    
    def read(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using attention weights.
        
        Args:
            weights: Attention weights (batch, num_slots)
        
        Returns:
            Read vector (batch, slot_size)
        """
        return torch.matmul(weights, self.memory.unsqueeze(0).expand(weights.size(0), -1, -1))
    
    def write(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        add: bool = True,
    ) -> None:
        """
        Write to memory.
        
        Args:
            values: Values to write (batch, slot_size)
            weights: Write weights (batch, num_slots)
            add: If True, add to existing memory; else overwrite
        """
        if add:
            # Additive write
            write_matrix = torch.matmul(
                weights.unsqueeze(-1),
                values.unsqueeze(1),
            )  # (batch, num_slots, slot_size)
            self.memory.data += write_matrix.mean(0)
        else:
            # Overwrite (using weighted average)
            write_matrix = torch.matmul(
                weights.unsqueeze(-1),
                values.unsqueeze(1),
            )
            self.memory.data = write_matrix.mean(0)
    
    def reset(self) -> None:
        """Reset memory to initial state."""
        nn.init.xavier_uniform_(self.memory)


class DifferentiableMemory(nn.Module):
    """
    Differentiable memory with content-based addressing.
    
    Implements content-based attention for reading and writing.
    
    Args:
        num_slots: Number of memory slots
        slot_size: Size of each memory slot
        num_reads: Number of read heads
        num_writes: Number of write heads
    """
    
    def __init__(
        self,
        num_slots: int = 256,
        slot_size: int = 128,
        num_reads: int = 4,
        num_writes: int = 1,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_reads = num_reads
        self.num_writes = num_writes
        
        # Memory
        self.memory = nn.Parameter(torch.zeros(num_slots, slot_size))
        nn.init.xavier_uniform_(self.memory)
        
        # Read and write heads
        self.read_weights = nn.Parameter(torch.zeros(1, num_reads, num_slots))
        self.write_weights = nn.Parameter(torch.zeros(1, num_writes, num_slots))
        
        # Initialize with uniform distribution
        nn.init.constant_(self.read_weights, 1.0 / num_slots)
        nn.init.constant_(self.write_weights, 1.0 / num_slots)
    
    def content_address(
        self,
        key: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Content-based addressing.
        
        Args:
            key: Key vector (batch, slot_size)
            strength: Key strength (sharpness)
        
        Returns:
            Attention weights (batch, num_slots)
        """
        # Cosine similarity
        memory_norm = F.normalize(self.memory.unsqueeze(0), dim=-1)
        key_norm = F.normalize(key.unsqueeze(1), dim=-1)
        
        similarity = torch.matmul(key_norm, memory_norm.transpose(-2, -1))
        weights = F.softmax(similarity * strength, dim=-1)
        
        return weights
    
    def read(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Read from memory.
        
        Args:
            weights: Read weights (batch, num_reads, num_slots)
        
        Returns:
            Read vectors (batch, num_reads, slot_size)
        """
        return torch.matmul(weights, self.memory.unsqueeze(0).expand(weights.size(0), -1, -1))
    
    def write(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        erase_vector: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Write to memory with erase and add.
        
        Args:
            values: Values to add (batch, num_writes, slot_size)
            weights: Write weights (batch, num_writes, num_slots)
            erase_vector: Erase vector (batch, num_writes, slot_size)
        """
        batch_size = values.shape[0]
        
        # Erase
        if erase_vector is not None:
            erase_matrix = torch.matmul(
                weights.transpose(1, 2),  # (batch, num_slots, num_writes)
                erase_vector,  # (batch, num_writes, slot_size)
            )  # (batch, num_slots, slot_size)
            erase_matrix = erase_matrix.mean(0)
            self.memory.data *= (1 - erase_matrix)
        
        # Add
        add_matrix = torch.matmul(
            weights.transpose(1, 2),
            values,
        )
        add_matrix = add_matrix.mean(0)
        self.memory.data += add_matrix
    
    def get_read_vectors(self) -> torch.Tensor:
        """Get current read vectors."""
        return self.read(self.read_weights.expand(-1, -1, -1))
    
    def reset(self) -> None:
        """Reset memory."""
        nn.init.xavier_uniform_(self.memory)
        nn.init.constant_(self.read_weights, 1.0 / self.num_slots)
        nn.init.constant_(self.write_weights, 1.0 / self.num_slots)
