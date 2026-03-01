"""
Compile module - PyTorch 2.0 compilation support.

Provides utilities for torch.compile() integration with proper tracing,
dynamic shape handling, and performance optimization.
"""

from torchagentic.compile.core import (
    CompileConfig,
    compile_model,
    compile_function,
    is_compiled,
    get_compilation_info,
)
from torchagentic.compile.optimizations import (
    optimize_for_inference,
    optimize_for_training,
    optimize_memory,
    optimize_speed,
)
from torchagentic.compile.tracing import (
    TraceConfig,
    trace_model,
    export_to_onnx,
    export_to_torchscript,
)

__all__ = [
    # Core
    "CompileConfig",
    "compile_model",
    "compile_function",
    "is_compiled",
    "get_compilation_info",
    # Optimizations
    "optimize_for_inference",
    "optimize_for_training",
    "optimize_memory",
    "optimize_speed",
    # Tracing
    "TraceConfig",
    "trace_model",
    "export_to_onnx",
    "export_to_torchscript",
]
