"""
Core compilation utilities for PyTorch 2.0+.

Provides torch.compile() wrappers with proper configuration
for reinforcement learning workloads.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional
import torch
import torch.nn as nn


# Check PyTorch version for compile support
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile") and torch.__version__ >= "2.0.0"

if TORCH_COMPILE_AVAILABLE:
    try:
        import torch._dynamo as dynamo
        DYNAMO_AVAILABLE = True
    except ImportError:
        DYNAMO_AVAILABLE = False
else:
    DYNAMO_AVAILABLE = False


@dataclass
class CompileConfig:
    """
    Configuration for torch.compile().
    
    Attributes:
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs')
        dynamic: Enable dynamic shapes support
        fullgraph: Compile entire graph (no Python fallback)
        backend: Backend to use ('inductor', 'cudagraphs', 'onnxrt', 'tvm', etc.)
        options: Additional backend-specific options
        disable: Disable compilation (useful for debugging)
    """
    mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = "default"
    dynamic: bool = False
    fullgraph: bool = False
    backend: str = "inductor"
    options: dict[str, Any] = field(default_factory=dict)
    disable: bool = False
    
    # RL-specific optimizations
    inplace_encoding: bool = True  # Allow inplace operations
    capture_scalar_outputs: bool = True  # Capture scalar tensors
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode,
            "dynamic": self.dynamic,
            "fullgraph": self.fullgraph,
            "backend": self.backend,
            "options": self.options,
            "disable": self.disable,
            "inplace_encoding": self.inplace_encoding,
            "capture_scalar_outputs": self.capture_scalar_outputs,
        }
    
    @classmethod
    def for_inference(cls) -> "CompileConfig":
        """Configuration optimized for inference."""
        return cls(
            mode="max-autotune",
            dynamic=False,
            fullgraph=True,
        )
    
    @classmethod
    def for_training(cls) -> "CompileConfig":
        """Configuration optimized for training."""
        return cls(
            mode="default",
            dynamic=True,
            fullgraph=False,
        )
    
    @classmethod
    def for_rl_inference(cls) -> "CompileConfig":
        """Configuration optimized for RL inference (low latency)."""
        return cls(
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=True,
            options={"trace.enabled": True},
        )
    
    @classmethod
    def for_rl_training(cls) -> "CompileConfig":
        """Configuration optimized for RL training."""
        return cls(
            mode="default",
            dynamic=True,
            fullgraph=False,
            options={
                "triton.cudagraphs": False,  # Disable for RL due to control flow
            },
        )


def compile_model(
    model: nn.Module,
    config: Optional[CompileConfig] = None,
    example_inputs: Optional[tuple[torch.Tensor, ...]] = None,
    warmup: bool = True,
) -> nn.Module:
    """
    Compile a PyTorch model with torch.compile().
    
    Args:
        model: Model to compile
        config: Compilation configuration
        example_inputs: Example inputs for warmup
        warmup: Run warmup iteration
    
    Returns:
        Compiled model
    
    Example:
        >>> from torchagentic import DQN, ModelConfig, compile_model
        >>> model = DQN(ModelConfig(input_dim=4, action_dim=2))
        >>> compiled = compile_model(model, mode="reduce-overhead")
    """
    if not TORCH_COMPILE_AVAILABLE:
        warnings.warn(
            "torch.compile() requires PyTorch 2.0+. "
            f"Current version: {torch.__version__}. "
            "Returning uncompiled model."
        )
        return model
    
    if config is None:
        config = CompileConfig()
    
    if config.disable:
        return model
    
    # Apply torch.compile
    compiled_model = torch.compile(
        model,
        mode=config.mode,
        dynamic=config.dynamic,
        fullgraph=config.fullgraph,
        backend=config.backend,
        options=config.options,
    )
    
    # Mark model as compiled
    compiled_model._torchagentic_compiled = True  # type: ignore
    compiled_model._torchagentic_config = config  # type: ignore
    
    # Warmup
    if warmup and example_inputs is not None:
        compiled_model.train()
        with torch.enable_grad():
            _ = compiled_model(*example_inputs)
        
        compiled_model.eval()
        with torch.inference_mode():
            _ = compiled_model(*example_inputs)
    
    return compiled_model


def compile_function(
    fn: Callable,
    config: Optional[CompileConfig] = None,
) -> Callable:
    """
    Compile a function with torch.compile().
    
    Args:
        fn: Function to compile
        config: Compilation configuration
    
    Returns:
        Compiled function
    """
    if not TORCH_COMPILE_AVAILABLE:
        warnings.warn(
            "torch.compile() requires PyTorch 2.0+. "
            f"Current version: {torch.__version__}. "
            "Returning uncompiled function."
        )
        return fn
    
    if config is None:
        config = CompileConfig()
    
    if config.disable:
        return fn
    
    return torch.compile(
        fn,
        mode=config.mode,
        dynamic=config.dynamic,
        fullgraph=config.fullgraph,
        backend=config.backend,
        options=config.options,
    )


def is_compiled(model: nn.Module) -> bool:
    """Check if a model has been compiled with torch.compile()."""
    return getattr(model, "_torchagentic_compiled", False)


def get_compilation_info(model: nn.Module) -> dict[str, Any]:
    """
    Get compilation information for a model.
    
    Args:
        model: Model to query
    
    Returns:
        Dictionary with compilation info
    """
    if not is_compiled(model):
        return {"compiled": False}
    
    config = getattr(model, "_torchagentic_config", None)
    
    return {
        "compiled": True,
        "config": config.to_dict() if config else {},
        "backend": getattr(model, "backend", "unknown"),
    }


class CompiledModuleMixin:
    """
    Mixin class for compiled module support.
    
    Add this mixin to model classes for built-in compilation support.
    """
    
    _compiled: bool = False
    _compile_config: Optional[CompileConfig] = None
    
    def compile(
        self,
        config: Optional[CompileConfig] = None,
        example_inputs: Optional[tuple[torch.Tensor, ...]] = None,
        warmup: bool = True,
    ) -> "CompiledModuleMixin":
        """
        Compile the model.
        
        Args:
            config: Compilation configuration
            example_inputs: Example inputs for warmup
            warmup: Run warmup iteration
        
        Returns:
            Self for method chaining
        """
        if TORCH_COMPILE_AVAILABLE and (config is None or not config.disable):
            config = config or CompileConfig()
            
            self._compiled = True
            self._compile_config = config
            
            # Recompile with new config
            compiled = torch.compile(
                self,
                mode=config.mode,
                dynamic=config.dynamic,
                fullgraph=config.fullgraph,
                backend=config.backend,
                options=config.options,
            )
            
            # Copy compiled state
            self.__dict__.update(compiled.__dict__)
            
            if warmup and example_inputs is not None:
                self.train()
                _ = self(*example_inputs)
                self.eval()
                _ = self(*example_inputs)
        
        return self
    
    @property
    def is_compiled(self) -> bool:
        """Check if model is compiled."""
        return self._compiled
    
    def reset_compile(self) -> None:
        """Reset compilation state."""
        self._compiled = False
        self._compile_config = None


def get_optimal_compile_config(
    model_type: str,
    device: str = "cuda",
    batch_size: int = 1,
) -> CompileConfig:
    """
    Get optimal compilation config for specific model types.
    
    Args:
        model_type: Type of model ('dqn', 'ppo', 'sac', 'transformer', etc.)
        device: Target device
        batch_size: Expected batch size
    
    Returns:
        Optimized CompileConfig
    """
    # RL-specific optimizations
    rl_models = {"dqn", "double_dqn", "dueling_dqn", "noisy_dqn"}
    policy_gradient = {"ppo", "a3c", "sac", "td3", "ddpg"}
    transformers = {"transformer", "decision_transformer", "perceiver"}
    
    if model_type.lower() in rl_models:
        # Value-based methods benefit from reduce-overhead
        return CompileConfig.for_rl_inference()
    
    elif model_type.lower() in policy_gradient:
        # Policy gradient needs gradient tracking
        if device == "cuda" and batch_size >= 32:
            return CompileConfig(
                mode="default",
                dynamic=True,
                options={"triton.cudagraphs": False},
            )
        return CompileConfig.for_rl_training()
    
    elif model_type.lower() in transformers:
        # Transformers benefit from max-autotune
        return CompileConfig(
            mode="max-autotune" if device == "cuda" else "default",
            dynamic=True,
        )
    
    else:
        # Default config
        return CompileConfig()
