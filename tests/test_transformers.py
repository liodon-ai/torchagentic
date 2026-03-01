"""
Tests for transformer models.
"""

import pytest
import torch
from torchagentic import TransformerAgent, DecisionTransformer, ModelConfig
from torchagentic.transformers.attention import SelfAttention, MultiHeadAttention, TransformerBlock


class TestSelfAttention:
    """Tests for SelfAttention."""
    
    def test_create_self_attention(self):
        attn = SelfAttention(embed_dim=64, num_heads=4)
        
        assert attn.num_heads == 4
    
    def test_self_attention_forward(self):
        attn = SelfAttention(embed_dim=64, num_heads=4)
        
        x = torch.randn(4, 10, 64)  # (batch, seq_len, embed_dim)
        out = attn(x)
        
        assert out.shape == (4, 10, 64)
    
    def test_self_attention_with_mask(self):
        attn = SelfAttention(embed_dim=64, num_heads=4)
        
        x = torch.randn(4, 10, 64)
        mask = torch.tril(torch.ones(10, 10))
        out = attn(x, mask)
        
        assert out.shape == (4, 10, 64)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""
    
    def test_create_multi_head_attention(self):
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        
        assert attn.num_heads == 4
    
    def test_multi_head_attention_forward(self):
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        
        q = torch.randn(4, 10, 64)
        k = torch.randn(4, 8, 64)
        v = torch.randn(4, 8, 64)
        
        out = attn(q, k, v)
        
        assert out.shape == (4, 10, 64)


class TestTransformerBlock:
    """Tests for TransformerBlock."""
    
    def test_create_transformer_block(self):
        block = TransformerBlock(embed_dim=64, num_heads=4)
        
        assert block is not None
    
    def test_transformer_block_forward(self):
        block = TransformerBlock(embed_dim=64, num_heads=4)
        
        x = torch.randn(4, 10, 64)
        out = block(x)
        
        assert out.shape == (4, 10, 64)


class TestTransformerAgent:
    """Tests for TransformerAgent."""
    
    def test_create_transformer_agent(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        agent = TransformerAgent(config, embed_dim=64, num_layers=2)
        
        assert agent.embed_dim == 64
    
    def test_transformer_agent_forward(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        agent = TransformerAgent(config, embed_dim=64, num_layers=2, max_seq_len=20)
        
        x = torch.randn(4, 10, 10)  # (batch, seq_len, input_dim)
        out = agent(x)
        
        assert out.shape == (4, 10, 64)
    
    def test_transformer_agent_get_action(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        agent = TransformerAgent(config, embed_dim=64, num_layers=2)
        
        x = torch.randn(4, 10, 10)
        action = agent.get_action(x)
        
        assert action.shape == (4,)


class TestDecisionTransformer:
    """Tests for DecisionTransformer."""
    
    def test_create_decision_transformer(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        dt = DecisionTransformer(config, embed_dim=64, max_seq_len=20)
        
        assert dt.max_seq_len == 20
    
    def test_decision_transformer_forward(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        dt = DecisionTransformer(config, embed_dim=64, max_seq_len=20)
        
        states = torch.randn(2, 10, 10)
        actions = torch.randn(2, 10, 4)
        rtgs = torch.ones(2, 10, 1) * 100
        
        predicted_actions = dt(states, actions, rtgs)
        
        assert predicted_actions.shape == (2, 10, 4)
    
    def test_decision_transformer_get_action(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        dt = DecisionTransformer(config, embed_dim=64, max_seq_len=20)
        
        state = torch.randn(1, 10)
        rtg = torch.tensor([[100.0]])
        
        action = dt.get_action(state, rtg)
        
        assert action.shape == (1, 4)
