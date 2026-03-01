"""
Tests for RL models.
"""

import pytest
import torch
from torchagentic import DQN, DuelingDQN, PPOActorCritic, ModelConfig


class TestDQN:
    """Tests for DQN."""
    
    def test_create_dqn(self):
        config = ModelConfig(input_dim=4, action_dim=2)
        dqn = DQN(config, image_input=False)
        
        assert dqn.config.action_dim == 2
    
    def test_dqn_forward(self):
        config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[32])
        dqn = DQN(config, image_input=False)
        
        x = torch.randn(4, 4)
        q_values = dqn(x)
        
        assert q_values.shape == (4, 2)
    
    def test_dqn_get_q_values(self):
        config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[32])
        dqn = DQN(config, image_input=False)
        
        x = torch.randn(4, 4)
        q_values = dqn.get_q_values(x)
        
        assert q_values.shape == (4, 2)
    
    def test_dqn_get_action(self):
        config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[32])
        dqn = DQN(config, image_input=False)
        
        x = torch.randn(4, 4)
        action = dqn.get_action(x, epsilon=0.1)
        
        assert action.shape == (4,)
        assert all(0 <= a < 2 for a in action.tolist())
    
    def test_dqn_get_value(self):
        config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[32])
        dqn = DQN(config, image_input=False)
        
        x = torch.randn(4, 4)
        value = dqn.get_value(x)
        
        assert value.shape == (4,)


class TestDuelingDQN:
    """Tests for DuelingDQN."""
    
    def test_create_dueling_dqn(self):
        config = ModelConfig(input_dim=4, action_dim=2)
        dqn = DuelingDQN(config, image_input=False)
        
        assert dqn is not None
    
    def test_dueling_dqn_forward(self):
        config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[32])
        dqn = DuelingDQN(config, image_input=False)
        
        x = torch.randn(4, 4)
        q_values = dqn(x)
        
        assert q_values.shape == (4, 2)
    
    def test_dueling_dqn_get_value(self):
        config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[32])
        dqn = DuelingDQN(config, image_input=False)
        
        x = torch.randn(4, 4)
        value = dqn.get_value(x)
        
        assert value.shape == (4,)


class TestPPOActorCritic:
    """Tests for PPOActorCritic."""
    
    def test_create_ppo_actor_critic(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 64])
        ppo = PPOActorCritic(config, continuous=False)
        
        assert ppo.continuous == False
    
    def test_ppo_forward(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        ppo = PPOActorCritic(config, continuous=False)
        
        x = torch.randn(4, 10)
        action_out, value = ppo(x)
        
        assert action_out.shape == (4, 4)
        assert value.shape == (4,)
    
    def test_ppo_get_action_and_value(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        ppo = PPOActorCritic(config, continuous=False)
        
        x = torch.randn(4, 10)
        action, log_prob, entropy, value = ppo.get_action_and_value(x)
        
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)
    
    def test_ppo_evaluate_actions(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        ppo = PPOActorCritic(config, continuous=False)
        
        x = torch.randn(4, 10)
        action = torch.randint(0, 4, (4,))
        
        log_prob, entropy, value = ppo.evaluate_actions(x, action)
        
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)
    
    def test_ppo_continuous(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        ppo = PPOActorCritic(config, continuous=True)
        
        x = torch.randn(4, 10)
        action, log_prob, entropy, value = ppo.get_action_and_value(x)
        
        assert action.shape == (4, 4)
        assert log_prob.shape == (4,)


class TestSAC:
    """Tests for SAC models."""
    
    def test_create_sac_actor(self):
        from torchagentic import SACActor
        
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 64])
        actor = SACActor(config)
        
        assert actor is not None
    
    def test_sac_actor_get_action(self):
        from torchagentic import SACActor
        
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        actor = SACActor(config)
        
        x = torch.randn(4, 10)
        action = actor.get_action(x)
        
        assert action.shape == (4, 4)
    
    def test_sac_critic(self):
        from torchagentic import SACCritic
        
        critic = SACCritic(input_dim=10, action_dim=4, hidden_dims=[64, 64])
        
        obs = torch.randn(4, 10)
        action = torch.randn(4, 4)
        q_value = critic(obs, action)
        
        assert q_value.shape == (4,)


class TestTD3:
    """Tests for TD3 models."""
    
    def test_create_td3_actor(self):
        from torchagentic import TD3Actor
        
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 64])
        actor = TD3Actor(config)
        
        assert actor is not None
    
    def test_td3_actor_get_action(self):
        from torchagentic import TD3Actor
        
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        actor = TD3Actor(config)
        
        x = torch.randn(4, 10)
        action = actor.get_action(x)
        
        assert action.shape == (4, 4)
        assert torch.all(action >= -1) and torch.all(action <= 1)
    
    def test_td3_critic(self):
        from torchagentic import TD3Critic
        
        critic = TD3Critic(input_dim=10, action_dim=4, hidden_dims=[64, 64])
        
        obs = torch.randn(4, 10)
        action = torch.randn(4, 4)
        q_value = critic(obs, action)
        
        assert q_value.shape == (4,)
