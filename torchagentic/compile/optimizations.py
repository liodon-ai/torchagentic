"""
Optimization presets for torch.compile().

Provides pre-configured optimization strategies for common use cases.
"""

from typing import Optional
import torch
import torch.nn as nn

from torchagentic.compile.core import CompileConfig, compile_model, TORCH_COMPILE_AVAILABLE


def optimize_for_inference(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 1,
    dynamic_shapes: bool = False,
) -> nn.Module:
    """
    Optimize model for inference.
    
    Applies aggressive optimizations for maximum inference speed.
    
    Args:
        model: Model to optimize
        device: Target device
        batch_size: Expected batch size
        dynamic_shapes: Allow dynamic batch sizes
    
    Returns:
        Optimized model
    """
    if not TORCH_COMPILE_AVAILABLE:
        return model
    
    config = CompileConfig(
        mode="max-autotune" if device == "cuda" else "default",
        dynamic=dynamic_shapes,
        fullgraph=True,
        backend="inductor",
        options={
            "triton.cudagraphs": device == "cuda",
            "max_autotune.gemm": True,
            "max_autotune.pointwise": True,
        },
    )
    
    # Example inputs for warmup
    if hasattr(model, "config"):
        example_inputs = _create_example_inputs(model, batch_size, device)
    else:
        example_inputs = None
    
    return compile_model(model, config, example_inputs, warmup=True)


def optimize_for_training(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 32,
    dynamic_shapes: bool = True,
) -> nn.Module:
    """
    Optimize model for training.
    
    Balances compilation time with training speed.
    
    Args:
        model: Model to optimize
        device: Target device
        batch_size: Expected batch size
        dynamic_shapes: Allow dynamic batch sizes
    
    Returns:
        Optimized model
    """
    if not TORCH_COMPILE_AVAILABLE:
        return model
    
    config = CompileConfig(
        mode="default",
        dynamic=dynamic_shapes,
        fullgraph=False,  # Allow Python fallback for gradients
        backend="inductor",
        options={
            "triton.cudagraphs": False,  # Disable for training
            "epilogue_fusion": True,
        },
    )
    
    example_inputs = _create_example_inputs(model, batch_size, device)
    return compile_model(model, config, example_inputs, warmup=True)


def optimize_memory(
    model: nn.Module,
    device: str = "cuda",
) -> nn.Module:
    """
    Optimize model for memory efficiency.
    
    Reduces memory usage at the cost of some speed.
    
    Args:
        model: Model to optimize
        device: Target device
    
    Returns:
        Optimized model
    """
    if not TORCH_COMPILE_AVAILABLE:
        return model
    
    config = CompileConfig(
        mode="default",
        dynamic=True,
        fullgraph=False,
        backend="inductor",
        options={
            "memory_planning.enabled": True,
            "memory_planning.memory_efficient_fusion": True,
            "activation_checkpointing": True,
        },
    )
    
    return compile_model(model, config, warmup=False)


def optimize_speed(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 1,
) -> nn.Module:
    """
    Optimize model for maximum speed.
    
    Uses aggressive autotuning for best performance.
    
    Args:
        model: Model to optimize
        device: Target device
        batch_size: Expected batch size
    
    Returns:
        Optimized model
    """
    if not TORCH_COMPILE_AVAILABLE:
        return model
    
    config = CompileConfig(
        mode="max-autotune",
        dynamic=False,
        fullgraph=True,
        backend="inductor",
        options={
            "max_autotune.gemm": True,
            "max_autotune.pointwise": True,
            "max_autotune.convolution": True,
            "triton.cudagraphs": device == "cuda",
        },
    )
    
    example_inputs = _create_example_inputs(model, batch_size, device)
    return compile_model(model, config, example_inputs, warmup=True)


def optimize_for_rl(
    model: nn.Module,
    model_type: str = "policy",
    device: str = "cuda",
) -> nn.Module:
    """
    Optimize model for RL workloads.
    
    Args:
        model: Model to optimize
        model_type: Type ('policy', 'value', 'q_network')
        device: Target device
    
    Returns:
        Optimized model
    """
    if not TORCH_COMPILE_AVAILABLE:
        return model
    
    if model_type == "policy":
        # Policy networks need low latency
        config = CompileConfig(
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=True,
            options={"trace.enabled": True},
        )
    elif model_type == "value":
        # Value networks can use more aggressive optimization
        config = CompileConfig(
            mode="max-autotune" if device == "cuda" else "default",
            dynamic=False,
            fullgraph=True,
        )
    else:  # q_network
        config = CompileConfig(
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=True,
        )
    
    return compile_model(model, config, warmup=True)


def _create_example_inputs(
    model: nn.Module,
    batch_size: int,
    device: str,
) -> Optional[tuple[torch.Tensor, ...]]:
    """Create example inputs for model warmup."""
    try:
        config = getattr(model, "config", None)
        if config is None:
            return None
        
        input_dim = getattr(config, "input_dim", 64)
        
        # Check if model expects image input
        image_input = getattr(model, "image_input", False)
        image_shape = getattr(model, "image_shape", None)
        
        if image_input and image_shape:
            # Image input
            x = torch.randn(batch_size, *image_shape, device=device)
        else:
            # Vector input
            x = torch.randn(batch_size, input_dim, device=device)
        
        return (x,)
    except Exception:
        return None


class OptimizationPreset:
    """
    Pre-defined optimization presets.
    
    Usage:
        preset = OptimizationPreset.inference()
        model = preset.apply(model)
    """
    
    @staticmethod
    def inference() -> CompileConfig:
        """Inference optimization."""
        return CompileConfig.for_inference()
    
    @staticmethod
    def training() -> CompileConfig:
        """Training optimization."""
        return CompileConfig.for_training()
    
    @staticmethod
    def rl_policy() -> CompileConfig:
        """RL policy network optimization."""
        return CompileConfig(
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=True,
        )
    
    @staticmethod
    def rl_value() -> CompileConfig:
        """RL value network optimization."""
        return CompileConfig(
            mode="max-autotune",
            dynamic=False,
            fullgraph=True,
        )
    
    @staticmethod
    def transformer() -> CompileConfig:
        """Transformer optimization."""
        return CompileConfig(
            mode="max-autotune",
            dynamic=True,
            options={
                "max_autotune.gemm": True,
                "max_autotune.pointwise": True,
            },
        )
    
    @staticmethod
    def memory_efficient() -> CompileConfig:
        """Memory-efficient optimization."""
        return CompileConfig(
            mode="default",
            dynamic=True,
            options={
                "memory_planning.enabled": True,
                "memory_planning.memory_efficient_fusion": True,
            },
        )
    
    @staticmethod
    def low_latency() -> CompileConfig:
        """Low-latency optimization."""
        return CompileConfig(
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=True,
        )
