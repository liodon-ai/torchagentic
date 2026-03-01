"""
TorchAgentic - PyTorch Model Definitions for AI Agents.

A library of neural network architectures for building trainable AI agents,
including RL agents, transformer-based models, memory-augmented networks,
and multi-agent systems.
"""

__version__ = "0.1.0"
__author__ = "Liodon AI"

# Core models
from torchagentic.models.base import BaseAgentModel, ModelConfig
from torchagentic.models.mlp import MLPNetwork
from torchagentic.models.cnn import CNNNetwork, NatureCNN, ResNetNetwork
from torchagentic.models.rnn import RNNNetwork, LSTMAgent, GRUAgent

# RL Agent Models
from torchagentic.rl.dqn import DQN, DuelingDQN, NoisyDQN
from torchagentic.rl.ppo import PPOActor, PPOCritic, PPOActorCritic
from torchagentic.rl.a3c import A3CNetwork
from torchagentic.rl.sac import SACActor, SACCritic, SACValue
from torchagentic.rl.td3 import TD3Actor, TD3Critic

# Transformer-based models
from torchagentic.transformers.agent import TransformerAgent, DecisionTransformer
from torchagentic.transformers.perceiver import PerceiverAgent
from torchagentic.transformers.attention import SelfAttention, MultiHeadAttention

# Memory-augmented networks
from torchagentic.memory.core import MemoryMatrix, DifferentiableMemory
from torchagentic.memory.ntm import NeuralTuringMachine
from torchagentic.memory.dnc import DifferentiableNeuralComputer

# Multi-agent architectures
from torchagentic.multiagent.base import MultiAgentBase
from torchagentic.multiagent.maddpg import MADDPGAgent
from torchagentic.multiagent.qmix import QMIXNetwork, VDNNetwork

# Utilities
from torchagentic.utils.initialization import orthogonal_init_, xavier_init_
from torchagentic.utils.normalization import RunningNorm, LayerNorm2D

# PyTorch 2.0 compilation support
try:
    from torchagentic.compile import (
        CompileConfig,
        compile_model,
        compile_function,
        optimize_for_inference,
        optimize_for_training,
        optimize_speed,
        optimize_memory,
        is_compiled,
    )
    COMPILE_SUPPORT = True
except ImportError:
    COMPILE_SUPPORT = False

__all__ = [
    # Base
    "BaseAgentModel",
    "ModelConfig",
    # Core architectures
    "MLPNetwork",
    "CNNNetwork",
    "NatureCNN",
    "ResNetNetwork",
    "RNNNetwork",
    "LSTMAgent",
    "GRUAgent",
    # RL Models
    "DQN",
    "DuelingDQN",
    "NoisyDQN",
    "PPOActor",
    "PPOCritic",
    "PPOActorCritic",
    "A3CNetwork",
    "SACActor",
    "SACCritic",
    "SACValue",
    "TD3Actor",
    "TD3Critic",
    # Transformers
    "TransformerAgent",
    "DecisionTransformer",
    "PerceiverAgent",
    "SelfAttention",
    "MultiHeadAttention",
    # Memory
    "MemoryMatrix",
    "DifferentiableMemory",
    "NeuralTuringMachine",
    "DifferentiableNeuralComputer",
    # Multi-agent
    "MultiAgentBase",
    "MADDPGAgent",
    "QMIXNetwork",
    "VDNNetwork",
    # Utils
    "orthogonal_init_",
    "xavier_init_",
    "RunningNorm",
    "LayerNorm2D",
    # Compile (if available)
    "CompileConfig",
    "compile_model",
    "compile_function",
    "optimize_for_inference",
    "optimize_for_training",
    "optimize_speed",
    "optimize_memory",
    "is_compiled",
    "COMPILE_SUPPORT",
]
