"""
Model tracing and export utilities.

Provides tools for tracing model execution and exporting to various formats.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from torchagentic.compile.core import TORCH_COMPILE_AVAILABLE


@dataclass
class TraceConfig:
    """
    Configuration for model tracing.
    
    Attributes:
        record_shapes: Record tensor shapes for profiling
        profile_memory: Profile memory usage
        with_stack: Include stack traces
        with_flops: Include FLOPs estimation
        with_modules: Include module hierarchy
    """
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = True
    with_modules: bool = True
    
    def to_torch_profiler_args(self) -> dict[str, Any]:
        """Convert to torch.profiler arguments."""
        return {
            "record_shapes": self.record_shapes,
            "profile_memory": self.profile_memory,
            "with_stack": self.with_stack,
        }


def trace_model(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    config: Optional[TraceConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """
    Trace model execution for profiling.
    
    Args:
        model: Model to trace
        inputs: Input tensors for tracing
        config: Trace configuration
        output_dir: Optional directory to save trace results
    
    Returns:
        Dictionary with trace results
    """
    config = config or TraceConfig()
    
    results = {
        "model_name": model.__class__.__name__,
        "input_shapes": [x.shape if hasattr(x, "shape") else None for x in inputs],
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    # Use torch.profiler if available
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            **config.to_torch_profiler_args(),
        ) as prof:
            model.eval()
            with torch.inference_mode():
                _ = model(*inputs)
        
        results["profiler"] = prof.key_averages().table(sort_by="cuda_time_total")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(output_path / "trace.json"))
            results["trace_file"] = str(output_path / "trace.json")
    
    except Exception as e:
        results["profiler_error"] = str(e)
    
    # Estimate FLOPs if requested
    if config.with_flops:
        try:
            flops = _estimate_flops(model, inputs)
            results["flops"] = flops
        except Exception:
            results["flops"] = None
    
    return results


def export_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shapes: tuple[tuple[int, ...], ...],
    opset_version: int = 17,
    dynamic_axes: Optional[dict[str, Any]] = None,
    device: str = "cpu",
) -> Path:
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export
        output_path: Path to save ONNX file
        input_shapes: Shapes of input tensors
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes specification
        device: Device for export
    
    Returns:
        Path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Create dummy inputs
    inputs = tuple(torch.randn(shape, device=device) for shape in input_shapes)
    
    # Export
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[f"input_{i}" for i in range(len(input_shapes))],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    
    return output_path


def export_to_torchscript(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shapes: Optional[tuple[tuple[int, ...], ...]] = None,
    method: str = "trace",
    device: str = "cpu",
) -> Path:
    """
    Export model to TorchScript format.
    
    Args:
        model: Model to export
        output_path: Path to save TorchScript file
        input_shapes: Shapes for tracing (required for trace method)
        method: Export method ('trace' or 'script')
        device: Device for export
    
    Returns:
        Path to exported TorchScript file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    if method == "trace":
        if input_shapes is None:
            raise ValueError("input_shapes required for trace method")
        
        inputs = tuple(torch.randn(shape, device=device) for shape in input_shapes)
        
        with torch.jit.optimized_execution(True):
            scripted = torch.jit.trace(model, inputs)
    else:  # script
        scripted = torch.jit.script(model)
    
    scripted.save(str(output_path))
    return output_path


def get_model_graph(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate model computation graph.
    
    Args:
        model: Model to visualize
        inputs: Input tensors
        output_path: Optional path to save graph
    
    Returns:
        Graph representation as string
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(output_path))
        else:
            import tempfile
            writer = SummaryWriter(tempfile.mkdtemp())
        
        writer.add_graph(model, inputs)
        writer.close()
        
        return f"Graph saved to {writer.log_dir}"
    
    except ImportError:
        warnings.warn("tensorboard not installed. Install with: pip install tensorboard")
        return "tensorboard required for graph visualization"


def benchmark_compilation(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict[str, float]:
    """
    Benchmark model compilation and execution.
    
    Args:
        model: Model to benchmark
        inputs: Input tensors
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
    
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    # Benchmark uncompiled
    model.eval()
    
    # Warmup
    for _ in range(num_warmup):
        with torch.inference_mode():
            _ = model(*inputs)
    
    # Time uncompiled
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start and end:
        start.record()
        for _ in range(num_runs):
            with torch.inference_mode():
                _ = model(*inputs)
        end.record()
        torch.cuda.synchronize()
        results["uncompiled_time_ms"] = start.elapsed_time(end) / num_runs
    else:
        import time
        start = time.perf_counter()
        for _ in range(num_runs):
            with torch.inference_mode():
                _ = model(*inputs)
        results["uncompiled_time_ms"] = (time.perf_counter() - start) * 1000 / num_runs
    
    # Compile model
    if TORCH_COMPILE_AVAILABLE:
        compiled_model = torch.compile(model, mode="default")
        
        # Warmup compiled
        for _ in range(num_warmup * 2):  # More warmup for compiled
            with torch.inference_mode():
                _ = compiled_model(*inputs)
        
        # Time compiled
        if start and end:
            start.record()
            for _ in range(num_runs):
                with torch.inference_mode():
                    _ = compiled_model(*inputs)
            end.record()
            torch.cuda.synchronize()
            results["compiled_time_ms"] = start.elapsed_time(end) / num_runs
        else:
            import time
            start = time.perf_counter()
            for _ in range(num_runs):
                with torch.inference_mode():
                    _ = compiled_model(*inputs)
            results["compiled_time_ms"] = (time.perf_counter() - start) * 1000 / num_runs
        
        # Calculate speedup
        results["speedup"] = results["uncompiled_time_ms"] / results["compiled_time_ms"]
        results["compilation_successful"] = True
    else:
        results["compilation_successful"] = False
        results["speedup"] = 1.0
    
    return results


def _estimate_flops(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> Optional[int]:
    """Estimate FLOPs for model forward pass."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        
        with FlopCounterMode() as flop_counter:
            model(*inputs)
        
        return flop_counter.get_total_flops()
    except Exception:
        return None
