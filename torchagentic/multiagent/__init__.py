"""
Multi-agent module - Multi-agent architectures.
"""

from torchagentic.multiagent.base import MultiAgentBase
from torchagentic.multiagent.maddpg import MADDPGAgent
from torchagentic.multiagent.qmix import QMIXNetwork, VDNNetwork

__all__ = [
    "MultiAgentBase",
    "MADDPGAgent",
    "QMIXNetwork",
    "VDNNetwork",
]
