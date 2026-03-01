"""
Memory module - Memory-augmented neural networks.
"""

from torchagentic.memory.core import MemoryMatrix, DifferentiableMemory
from torchagentic.memory.ntm import NeuralTuringMachine
from torchagentic.memory.dnc import DifferentiableNeuralComputer

__all__ = [
    "MemoryMatrix",
    "DifferentiableMemory",
    "NeuralTuringMachine",
    "DifferentiableNeuralComputer",
]
