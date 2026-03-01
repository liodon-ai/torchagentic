"""
RL module - Reinforcement Learning agent models.
"""

from torchagentic.rl.dqn import DQN, DuelingDQN, NoisyDQN
from torchagentic.rl.ppo import PPOActor, PPOCritic, PPOActorCritic
from torchagentic.rl.a3c import A3CNetwork
from torchagentic.rl.sac import SACActor, SACCritic, SACValue
from torchagentic.rl.td3 import TD3Actor, TD3Critic

__all__ = [
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
]
