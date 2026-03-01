"""
Tools module for TorchAgentic.

Provides the tool system for agents, including tool registration,
execution, and schema generation.
"""

from torchagentic.tools.base import Tool, ToolRegistry, tool
from torchagentic.tools.manager import ToolManager
from torchagentic.tools.builtin import (
    CalculatorTool,
    SearchTool,
    DateTimeTool,
    PythonREPLTool,
    FileReadTool,
    FileWriteTool,
)

__all__ = [
    "Tool",
    "ToolRegistry",
    "tool",
    "ToolManager",
    "CalculatorTool",
    "SearchTool",
    "DateTimeTool",
    "PythonREPLTool",
    "FileReadTool",
    "FileWriteTool",
]
