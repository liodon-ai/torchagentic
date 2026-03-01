"""
PyTorch 2.0 Compilation Example

This example demonstrates how to use torch.compile() with TorchAgentic models
for improved performance.

Requirements:
    - PyTorch 2.0 or later
    - CUDA GPU (recommended for best speedups)
"""

import time
import torch
from torchagentic import (
    DQN,
    PPOActorCritic,
    MLPNetwork,
    ModelConfig,
    compile_model,
    CompileConfig,
    optimize_for_inference,
    optimize_for_training,
    COMPILE_SUPPORT,
)


def benchmark_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    num_warmup: int = 50,
    num_runs: int = 200,
    name: str = "model",
) -> float:
    """Benchmark model inference latency."""
    model.eval()
    
    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(inputs)
    
    # Benchmark
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.inference_mode():
            for _ in range(num_runs):
                _ = model(inputs)
        end_event.record()
        torch.cuda.synchronize()
        
        avg_time_ms = start_event.elapsed_time(end_event) / num_runs
    else:
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(num_runs):
                _ = model(inputs)
        avg_time_ms = (time.perf_counter() - start) * 1000 / num_runs
    
    print(f"{name}: {avg_time_ms:.3f} ms/op")
    return avg_time_ms


def benchmark_training(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 50,
    name: str = "model",
) -> float:
    """Benchmark model training step."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Warmup
    for _ in range(num_warmup):
        optimizer.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_runs):
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
        end_event.record()
        torch.cuda.synchronize()
        
        avg_time_ms = start_event.elapsed_time(end_event) / num_runs
    else:
        start = time.perf_counter()
        for _ in range(num_runs):
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
        avg_time_ms = (time.perf_counter() - start) * 1000 / num_runs
    
    print(f"{name} (training): {avg_time_ms:.3f} ms/step")
    return avg_time_ms


def example_1_basic_compile():
    """Example 1: Basic compilation with default settings."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Compilation")
    print("=" * 60)
    
    if not COMPILE_SUPPORT:
        print("torch.compile() not available. Requires PyTorch 2.0+")
        return
    
    # Create model
    config = ModelConfig(input_dim=64, action_dim=4, hidden_dims=[256, 256])
    model = MLPNetwork(config)
    
    print(f"Original model: {model}")
    
    # Compile with default settings
    compiled_model = model.compile(mode="default")
    
    print(f"Compiled model: {compiled_model}")
    print(f"Is compiled: {compiled_model.is_compiled}")
    
    # Benchmark
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.randn(32, 64, device=device)
    
    print("\nInference benchmark (batch_size=32):")
    benchmark_inference(model.to(device), inputs, name="Uncompiled")
    benchmark_inference(compiled_model.to(device), inputs, name="Compiled")


def example_2_rl_models():
    """Example 2: Compiling RL models."""
    print("\n" + "=" * 60)
    print("Example 2: RL Model Compilation")
    print("=" * 60)
    
    if not COMPILE_SUPPORT:
        print("torch.compile() not available.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DQN for Atari
    print("\n--- DQN (Atari) ---")
    dqn = DQN(
        config=ModelConfig(input_dim=4, action_dim=6),
        image_input=True,
    )
    
    # Compile for inference (low latency)
    dqn.compile(mode="reduce-overhead", dynamic=False)
    
    obs = torch.randn(1, 4, 84, 84, device=device)
    print("DQN inference (single frame):")
    benchmark_inference(dqn.to(device), obs, num_runs=100, name="DQN Compiled")
    
    # PPO Actor-Critic
    print("\n--- PPO Actor-Critic ---")
    ppo = PPOActorCritic(
        config=ModelConfig(input_dim=24, action_dim=4, hidden_dims=[256, 256]),
        continuous=True,
    )
    
    # Compile for training
    ppo.compile(mode="default", dynamic=True)
    
    obs = torch.randn(128, 24, device=device)
    targets = torch.randn(128, 4, device=device)
    
    print("PPO training step (batch_size=128):")
    benchmark_training(ppo.to(device), obs, targets, num_runs=20, name="PPO Compiled")


def example_3_optimization_presets():
    """Example 3: Using optimization presets."""
    print("\n" + "=" * 60)
    print("Example 3: Optimization Presets")
    print("=" * 60)
    
    if not COMPILE_SUPPORT:
        print("torch.compile() not available.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig(input_dim=128, action_dim=8, hidden_dims=[512, 512])
    
    # Create multiple versions
    model_base = MLPNetwork(config).to(device)
    model_speed = MLPNetwork(config).to(device)
    model_memory = MLPNetwork(config).to(device)
    model_inference = MLPNetwork(config).to(device)
    
    # Apply different optimizations
    print("Applying optimizations...")
    model_speed.compile(mode="max-autotune")
    model_memory.compile(mode="default", options={"memory_planning.enabled": True})
    model_inference.compile(mode="reduce-overhead", fullgraph=True)
    
    inputs = torch.randn(64, 128, device=device)
    
    print("\nComparison (batch_size=64):")
    benchmark_inference(model_base, inputs, name="Base")
    benchmark_inference(model_speed, inputs, name="Speed Optimized")
    benchmark_inference(model_memory, inputs, name="Memory Optimized")
    benchmark_inference(model_inference, inputs, name="Inference Optimized")


def example_4_compile_config():
    """Example 4: Advanced CompileConfig usage."""
    print("\n" + "=" * 60)
    print("Example 4: Advanced CompileConfig")
    print("=" * 60)
    
    if not COMPILE_SUPPORT:
        print("torch.compile() not available.")
        return
    
    # Create custom config
    config = CompileConfig(
        mode="max-autotune",
        dynamic=False,
        fullgraph=True,
        backend="inductor",
        options={
            "triton.cudagraphs": torch.cuda.is_available(),
            "max_autotune.gemm": True,
        },
    )
    
    print(f"CompileConfig: {config.to_dict()}")
    
    # Apply to model
    model = MLPNetwork(ModelConfig(input_dim=64, action_dim=4))
    compiled = compile_model(model, config=config)
    
    print(f"Compiled: {compiled.is_compiled}")


def example_5_full_training_loop():
    """Example 5: Complete training loop with compilation."""
    print("\n" + "=" * 60)
    print("Example 5: Training Loop with Compilation")
    print("=" * 60)
    
    if not COMPILE_SUPPORT:
        print("torch.compile() not available.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create and compile model
    config = ModelConfig(input_dim=32, action_dim=4, hidden_dims=[256, 256])
    model = PPOActorCritic(config, continuous=False).to(device)
    model.compile(mode="default", dynamic=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Training loop
    print("\nRunning training steps...")
    for step in range(100):
        # Generate random data
        obs = torch.randn(64, 32, device=device)
        actions = torch.randint(0, 4, (64,), device=device)
        advantages = torch.randn(64, device=device)
        returns = torch.randn(64, device=device)
        
        # PPO update
        optimizer.zero_grad()
        
        log_probs, entropies, values = model.evaluate_actions(obs, actions)
        
        # Policy loss
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = ratio.clamp(0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = (values - returns).pow(2).mean()
        
        # Total loss
        loss = policy_loss + value_loss * 0.5 - entropies.mean() * 0.01
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")
    
    print("Training completed!")
    print(f"Final model: {model}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("TorchAgentic - PyTorch 2.0 Compilation Examples")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Compile support: {COMPILE_SUPPORT}")
    
    example_1_basic_compile()
    example_2_rl_models()
    example_3_optimization_presets()
    example_4_compile_config()
    example_5_full_training_loop()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
