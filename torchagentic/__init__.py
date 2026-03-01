"""
TorchAgentic - A PyTorch-based library for building AI agents and agentic workflows.

This library provides a comprehensive framework for creating, managing, and orchestrating
AI agents with support for tool calling, memory management, and multi-agent workflows.
"""

__version__ = "0.1.0"
__author__ = "Liodon AI"

from torchagentic.core.agent import Agent, BaseAgent
from torchagentic.core.message import Message, MessageRole, Conversation
from torchagentic.core.response import AgentResponse, ResponseStatus
from torchagentic.tools.base import Tool, ToolRegistry, tool
from torchagentic.tools.manager import ToolManager
from torchagentic.memory.base import Memory, BaseMemory
from torchagentic.memory.short_term import ShortTermMemory
from torchagentic.memory.long_term import LongTermMemory
from torchagentic.workflows.workflow import Workflow, Task
from torchagentic.workflows.orchestrator import Orchestrator
from torchagentic.llms.base import BaseLLM, LLMConfig
from torchagentic.llms.local import LocalLLM
from torchagentic.utils.config import Config
from torchagentic.utils.logging import setup_logging, get_logger

__all__ = [
    # Core
    "Agent",
    "BaseAgent",
    "Message",
    "MessageRole",
    "Conversation",
    "AgentResponse",
    "ResponseStatus",
    # Tools
    "Tool",
    "ToolRegistry",
    "tool",
    "ToolManager",
    # Memory
    "Memory",
    "BaseMemory",
    "ShortTermMemory",
    "LongTermMemory",
    # Workflows
    "Workflow",
    "Task",
    "Orchestrator",
    # LLMs
    "BaseLLM",
    "LLMConfig",
    "LocalLLM",
    # Utils
    "Config",
    "setup_logging",
    "get_logger",
]
