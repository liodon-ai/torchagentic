"""
Tests for PyTorch 2.0 compilation support.
"""

import pytest
import torch
from torchagentic import (
    MLPNetwork,
    DQN,
    PPOActorCritic,
    ModelConfig,
    CompileConfig,
    compile_model,
    is_compiled,
    COMPILE_SUPPORT,
)


# Skip all tests if torch.compile not available
pytestmark = pytest.mark.skipif(
    not COMPILE_SUPPORT,
    reason="torch.compile() requires PyTorch 2.0+",
)


class TestCompileConfig:
    """Tests for CompileConfig."""
    
    def test_create_default_config(self):
        config = CompileConfig()
        
        assert config.mode == "default"
        assert config.dynamic == False
        assert config.fullgraph == False
        assert config.backend == "inductor"
    
    def test_create_inference_config(self):
        config = CompileConfig.for_inference()
        
        assert config.mode == "max-autotune"
        assert config.fullgraph == True
    
    def test_create_training_config(self):
        config = CompileConfig.for_training()
        
        assert config.mode == "default"
        assert config.dynamic == True
    
    def test_create_rl_inference_config(self):
        config = CompileConfig.for_rl_inference()
        
        assert config.mode == "reduce-overhead"
        assert config.fullgraph == True
    
    def test_config_to_dict(self):
        config = CompileConfig(mode="max-autotune", dynamic=True)
        data = config.to_dict()
        
        assert data["mode"] == "max-autotune"
        assert data["dynamic"] == True


class TestCompileModel:
    """Tests for compile_model function."""
    
    def test_compile_mlp(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[32])
        model = MLPNetwork(config)
        
        compiled = compile_model(model, config=CompileConfig(disable=False))
        
        assert is_compiled(compiled)
    
    def test_compile_dqn(self):
        config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[32])
        model = DQN(config, image_input=False)
        
        compiled = compile_model(model, warmup=False)
        
        assert is_compiled(compiled)
    
    def test_compile_with_warmup(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[32])
        model = MLPNetwork(config)
        
        example_inputs = (torch.randn(4, 10),)
        compiled = compile_model(model, example_inputs=example_inputs, warmup=True)
        
        # Should run without errors after warmup
        with torch.inference_mode():
            _ = compiled(torch.randn(4, 10))
    
    def test_compile_disabled(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        config = CompileConfig(disable=True)
        compiled = compile_model(model, config=config)
        
        assert not is_compiled(compiled)


class TestModelCompileMethod:
    """Tests for model.compile() method."""
    
    def test_mlp_compile_method(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[32])
        model = MLPNetwork(config)
        
        result = model.compile(mode="default", warmup=False)
        
        assert result is model  # Returns self
        assert model.is_compiled
    
    def test_dqn_compile_method(self):
        config = ModelConfig(input_dim=4, action_dim=2)
        model = DQN(config, image_input=False)
        
        model.compile(mode="reduce-overhead", warmup=False)
        
        assert model.is_compiled
    
    def test_ppo_compile_method(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        model = PPOActorCritic(config, continuous=False)
        
        model.compile(mode="default", dynamic=True, warmup=False)
        
        assert model.is_compiled
    
    def test_compile_chain(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        # Should support method chaining
        result = model.compile(mode="default").compile(mode="reduce-overhead")
        
        assert result is model
    
    def test_reset_compile(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        model.compile(mode="default", warmup=False)
        assert model.is_compiled
        
        model.reset_compile()
        assert not model.is_compiled


class TestCompiledModelExecution:
    """Tests for executing compiled models."""
    
    def test_compiled_mlp_forward(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[32])
        model = MLPNetwork(config)
        model.compile(mode="default", warmup=True, example_inputs=(torch.randn(4, 10),))
        
        x = torch.randn(4, 10)
        with torch.inference_mode():
            out = model(x)
        
        assert out.shape == (4, 4)
    
    def test_compiled_mlp_get_action(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[32])
        model = MLPNetwork(config)
        model.compile(mode="reduce-overhead", warmup=True)
        
        x = torch.randn(4, 10)
        with torch.inference_mode():
            action = model.get_action(x)
        
        assert action.shape == (4,)
    
    def test_compiled_mlp_get_value(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[32])
        model = MLPNetwork(config)
        model.compile(mode="default", warmup=True)
        
        x = torch.randn(4, 10)
        with torch.inference_mode():
            value = model.get_value(x)
        
        assert value.shape == (4,)
    
    def test_compiled_dqn_inference(self):
        config = ModelConfig(input_dim=4, action_dim=2, hidden_dims=[32])
        model = DQN(config, image_input=False)
        model.compile(mode="reduce-overhead", warmup=True)
        
        x = torch.randn(4, 4)
        with torch.inference_mode():
            q_values = model.get_q_values(x)
        
        assert q_values.shape == (4, 2)
    
    def test_compiled_ppo_action_and_value(self):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[32])
        model = PPOActorCritic(config, continuous=False)
        model.compile(mode="default", warmup=True)
        
        x = torch.randn(4, 10)
        with torch.inference_mode():
            action, log_prob, entropy, value = model.get_action_and_value(x)
        
        assert action.shape == (4,)
        assert value.shape == (4,)


class TestCompileModes:
    """Tests for different compilation modes."""
    
    @pytest.mark.parametrize("mode", ["default", "reduce-overhead", "max-autotune"])
    def test_compile_modes(self, mode):
        config = ModelConfig(input_dim=10, action_dim=4, hidden_dims=[32])
        model = MLPNetwork(config)
        
        model.compile(mode=mode, warmup=False)
        
        assert model.is_compiled
    
    def test_dynamic_shapes(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        model.compile(dynamic=True, warmup=False)
        
        # Should handle different batch sizes
        with torch.inference_mode():
            _ = model(torch.randn(1, 10))
            _ = model(torch.randn(16, 10))
            _ = model(torch.randn(64, 10))
    
    def test_fullgraph_compilation(self):
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        model.compile(fullgraph=True, warmup=False)
        
        assert model.is_compiled


class TestOptimizationFunctions:
    """Tests for optimization helper functions."""
    
    def test_optimize_for_inference(self):
        from torchagentic import optimize_for_inference
        
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        optimized = optimize_for_inference(model, device="cpu")
        
        assert is_compiled(optimized)
    
    def test_optimize_for_training(self):
        from torchagentic import optimize_for_training
        
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        optimized = optimize_for_training(model, device="cpu", batch_size=32)
        
        assert is_compiled(optimized)
    
    def test_optimize_speed(self):
        from torchagentic import optimize_speed
        
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        optimized = optimize_speed(model, device="cpu")
        
        assert is_compiled(optimized)
    
    def test_optimize_memory(self):
        from torchagentic import optimize_memory
        
        config = ModelConfig(input_dim=10, action_dim=4)
        model = MLPNetwork(config)
        
        optimized = optimize_memory(model, device="cpu")
        
        assert is_compiled(optimized)
