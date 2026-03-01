"""
Tests for core models.
"""

import pytest
import torch
from torchagentic import MLPNetwork, CNNNetwork, NatureCNN, RNNNetwork, LSTMAgent, GRUAgent, ModelConfig


class TestMLPNetwork:
    """Tests for MLPNetwork."""
    
    def test_create_mlp(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 32])
        mlp = MLPNetwork(config)
        
        assert mlp.config.input_dim == 10
        assert mlp.config.action_dim == 4
    
    def test_mlp_forward(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 32])
        mlp = MLPNetwork(config)
        
        x = torch.randn(8, 10)
        out = mlp(x)
        
        assert out.shape == (8, 32)
    
    def test_mlp_get_action(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 32])
        mlp = MLPNetwork(config)
        
        x = torch.randn(8, 10)
        action = mlp.get_action(x)
        
        assert action.shape == (8,)
    
    def test_mlp_get_value(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 32])
        mlp = MLPNetwork(config)
        
        x = torch.randn(8, 10)
        value = mlp.get_value(x)
        
        assert value.shape == (8,)
    
    def test_mlp_action_and_value(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 32])
        mlp = MLPNetwork(config)
        
        x = torch.randn(8, 10)
        action, value = mlp.get_action_and_value(x)
        
        assert action.shape == (8,)
        assert value.shape == (8,)
    
    def test_mlp_parameter_count(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64, 32])
        mlp = MLPNetwork(config)
        
        assert mlp.get_num_params() > 0
        assert mlp.get_trainable_params() > 0


class TestCNNNetwork:
    """Tests for CNNNetwork."""
    
    def test_create_cnn(self):
        config = ModelConfig(input_dim=4, action_dim=6)  # 4 channels (stacked frames)
        cnn = CNNNetwork(config, image_shape=(4, 84, 84))
        
        assert cnn.image_shape == (4, 84, 84)
    
    def test_cnn_forward(self):
        config = ModelConfig(input_dim=4, action_dim=6)
        cnn = CNNNetwork(config, image_shape=(4, 84, 84))
        
        x = torch.randn(4, 4, 84, 84)  # (B, C, H, W)
        out = cnn(x)
        
        assert out.shape == (4, 512)
    
    def test_cnn_get_action(self):
        config = ModelConfig(input_dim=4, action_dim=6)
        cnn = CNNNetwork(config, image_shape=(4, 84, 84))
        
        x = torch.randn(4, 4, 84, 84)
        action = cnn.get_action(x)
        
        assert action.shape == (4,)


class TestNatureCNN:
    """Tests for NatureCNN."""
    
    def test_create_nature_cnn(self):
        config = ModelConfig(input_dim=4, action_dim=6)
        cnn = NatureCNN(config, image_shape=(4, 84, 84))
        
        assert cnn.image_shape == (4, 84, 84)
    
    def test_nature_cnn_forward(self):
        config = ModelConfig(input_dim=4, action_dim=6)
        cnn = NatureCNN(config, image_shape=(4, 84, 84))
        
        x = torch.randn(4, 4, 84, 84)
        out = cnn(x)
        
        assert out.shape == (4, 512)
    
    def test_nature_cnn_q_values(self):
        config = ModelConfig(input_dim=4, action_dim=6)
        cnn = NatureCNN(config, image_shape=(4, 84, 84))
        
        x = torch.randn(4, 4, 84, 84)
        q_values = cnn.get_q_values(x)
        
        assert q_values.shape == (4, 6)


class TestRNNNetwork:
    """Tests for RNNNetwork."""
    
    def test_create_rnn(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        rnn = RNNNetwork(config, rnn_type="lstm", hidden_size=128)
        
        assert rnn.hidden_size == 128
    
    def test_rnn_forward(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        rnn = RNNNetwork(config, rnn_type="lstm", hidden_size=128)
        
        x = torch.randn(4, 20, 10)  # (batch, seq_len, input_dim)
        out, hidden = rnn(x)
        
        assert out.shape == (4, 128)
    
    def test_rnn_get_action(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        rnn = RNNNetwork(config, rnn_type="lstm", hidden_size=128)
        
        x = torch.randn(4, 10)
        action = rnn.get_action(x)
        
        assert action.shape == (4,)
    
    def test_rnn_reset(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[64])
        rnn = RNNNetwork(config, rnn_type="lstm", hidden_size=128)
        
        rnn._hidden = (torch.randn(1, 1, 128), torch.randn(1, 1, 128))
        rnn.reset()
        
        assert rnn._hidden is None


class TestLSTMAgent:
    """Tests for LSTMAgent."""
    
    def test_create_lstm_agent(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        lstm = LSTMAgent(config, hidden_size=128)
        
        assert lstm.hidden_size == 128
    
    def test_lstm_get_action_and_value(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        lstm = LSTMAgent(config, hidden_size=128)
        
        x = torch.randn(4, 10)
        action, value = lstm.get_action_and_value(x)
        
        assert action.shape == (4,)
        assert value.shape == (4,)


class TestGRUAgent:
    """Tests for GRUAgent."""
    
    def test_create_gru_agent(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        gru = GRUAgent(config, hidden_size=128)
        
        assert gru.hidden_size == 128
    
    def test_gru_get_action(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        gru = GRUAgent(config, hidden_size=128)
        
        x = torch.randn(4, 10)
        action = gru.get_action(x)
        
        assert action.shape == (4,)
