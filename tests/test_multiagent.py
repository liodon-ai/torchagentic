"""
Tests for multi-agent models.
"""

import pytest
import torch
from torchagentic import MADDPGAgent, QMIXNetwork, VDNNetwork


class TestMADDPGAgent:
    """Tests for MADDPGAgent."""
    
    def test_create_maddpg(self):
        model = MADDPGAgent(
            num_agents=3,
            obs_dim=10,
            action_dim=2,
            hidden_dims=[64, 64],
        )
        
        assert model.num_agents == 3
    
    def test_maddpg_forward(self):
        model = MADDPGAgent(
            num_agents=3,
            obs_dim=10,
            action_dim=2,
            hidden_dims=[64],
        )
        
        observations = torch.randn(4, 3, 10)  # (batch, num_agents, obs_dim)
        actions = model(observations)
        
        assert actions.shape == (4, 3, 2)
    
    def test_maddpg_get_actions(self):
        model = MADDPGAgent(
            num_agents=3,
            obs_dim=10,
            action_dim=2,
            hidden_dims=[64],
        )
        
        observations = torch.randn(4, 3, 10)
        actions = model.get_actions(observations)
        
        assert actions.shape == (4, 3, 2)
    
    def test_maddpg_get_q_value(self):
        model = MADDPGAgent(
            num_agents=3,
            obs_dim=10,
            action_dim=2,
            hidden_dims=[64],
        )
        
        observations = torch.randn(4, 3, 10)
        actions = torch.randn(4, 3, 2)
        
        q_value = model.get_q_value(observations, actions)
        
        assert q_value.shape == (4,)
    
    def test_maddpg_shared_vs_individual_params(self):
        # Shared params
        model_shared = MADDPGAgent(
            num_agents=3,
            obs_dim=10,
            action_dim=2,
            shared_params=True,
        )
        
        # Individual params
        model_individual = MADDPGAgent(
            num_agents=3,
            obs_dim=10,
            action_dim=2,
            shared_params=False,
        )
        
        # Shared should have fewer parameters
        assert model_shared.get_num_params() < model_individual.get_num_params()


class TestQMIXNetwork:
    """Tests for QMIXNetwork."""
    
    def test_create_qmix(self):
        model = QMIXNetwork(
            num_agents=3,
            obs_dim=10,
            action_dim=4,
            hidden_dim=64,
        )
        
        assert model.num_agents == 3
    
    def test_qmix_forward(self):
        model = QMIXNetwork(
            num_agents=3,
            obs_dim=10,
            action_dim=4,
            hidden_dim=64,
        )
        
        observations = torch.randn(4, 3, 10)
        actions = torch.zeros(4, 3, 4)  # One-hot actions
        actions[:, :, 0] = 1  # All agents take action 0
        global_state = torch.randn(4, 32)
        
        q_total = model(observations, actions, global_state)
        
        assert q_total.shape == (4,)
    
    def test_qmix_get_actions(self):
        model = QMIXNetwork(
            num_agents=3,
            obs_dim=10,
            action_dim=4,
            hidden_dim=64,
        )
        
        observations = torch.randn(4, 3, 10)
        actions = model.get_actions(observations)
        
        assert actions.shape == (4, 3)
    
    def test_qmix_get_individual_q_values(self):
        model = QMIXNetwork(
            num_agents=3,
            obs_dim=10,
            action_dim=4,
            hidden_dim=64,
        )
        
        observations = torch.randn(4, 3, 10)
        q_values = model.get_individual_q_values(observations)
        
        assert q_values.shape == (4, 3, 4)


class TestVDNNetwork:
    """Tests for VDNNetwork."""
    
    def test_create_vdn(self):
        model = VDNNetwork(
            num_agents=3,
            obs_dim=10,
            action_dim=4,
            hidden_dim=64,
        )
        
        assert model.num_agents == 3
    
    def test_vdn_forward(self):
        model = VDNNetwork(
            num_agents=3,
            obs_dim=10,
            action_dim=4,
            hidden_dim=64,
        )
        
        observations = torch.randn(4, 3, 10)
        actions = torch.zeros(4, 3, 4)
        actions[:, :, 0] = 1
        
        q_total = model(observations, actions)
        
        assert q_total.shape == (4,)
    
    def test_vdn_get_individual_q_values(self):
        model = VDNNetwork(
            num_agents=3,
            obs_dim=10,
            action_dim=4,
            hidden_dim=64,
        )
        
        observations = torch.randn(4, 3, 10)
        q_values = model.get_individual_q_values(observations)
        
        assert q_values.shape == (4, 3, 4)
