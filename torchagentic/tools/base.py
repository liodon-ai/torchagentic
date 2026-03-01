"""
Base tool classes and decorators for TorchAgentic.

This module provides the foundation for creating tools that agents can use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints, get_origin, get_args
import inspect
import uuid
import functools


@dataclass
class ToolParameter:
    """Represents a parameter for a tool."""
    
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list[Any]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for LLM schemas."""
        result = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            result["enum"] = self.enum
        return result


@dataclass
class ToolDefinition:
    """Complete definition of a tool including metadata and schema."""
    
    name: str
    description: str
    function: Callable
    parameters: list[ToolParameter] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: p.to_dict() for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
            "tags": self.tags,
        }
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p.name: p.to_dict() for p in self.parameters
                    },
                    "required": [p.name for p in self.parameters if p.required],
                },
            },
        }


def _python_type_to_json_type(py_type: Any) -> str:
    """Convert Python type to JSON schema type."""
    if py_type is str:
        return "string"
    elif py_type is int:
        return "integer"
    elif py_type is float:
        return "number"
    elif py_type is bool:
        return "boolean"
    elif py_type is list or get_origin(py_type) is list:
        return "array"
    elif py_type is dict or get_origin(py_type) is dict:
        return "object"
    elif py_type is type(None):
        return "null"
    else:
        return "string"  # Default to string


def _extract_parameters(func: Callable) -> list[ToolParameter]:
    """Extract parameters from a function signature."""
    params = []
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    # Get parameter descriptions from docstring
    docstring = inspect.getdoc(func) or ""
    param_descriptions = {}
    
    for line in docstring.split("\n"):
        line = line.strip()
        if line.startswith("Args:"):
            continue
        if ":" in line and ("    " in line or line.startswith("    ")):
            parts = line.split(":", 1)
            if len(parts) == 2:
                param_name = parts[0].strip().lstrip("args:").lstrip("kwargs:").strip()
                param_desc = parts[1].strip()
                if param_name and param_desc:
                    param_descriptions[param_name] = param_desc
    
    for name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        
        # Get type
        py_type = type_hints.get(name, param.annotation)
        if py_type is inspect.Parameter.empty:
            json_type = "string"
        else:
            json_type = _python_type_to_json_type(py_type)
        
        # Check if required
        required = param.default is inspect.Parameter.empty
        
        params.append(ToolParameter(
            name=name,
            type=json_type,
            description=param_descriptions.get(name, f"The {name} parameter"),
            required=required,
            default=param.default if not required else None,
        ))
    
    return params


class Tool(ABC):
    """
    Abstract base class for tools.
    
    Subclasses should implement the `execute` method.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ):
        self._name = name or self.__class__.__name__
        self._description = description or self.__doc__ or ""
        self._tags = tags or []
        self._id: str = str(uuid.uuid4())
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def tags(self) -> list[str]:
        return self._tags
    
    @property
    def id(self) -> str:
        return self._id
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given arguments."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Get the parameters for this tool."""
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Convert tool to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p.name: p.to_dict() for p in self.get_parameters()
                },
                "required": [p.name for p in self.get_parameters() if p.required],
            },
            "tags": self.tags,
        }
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p.name: p.to_dict() for p in self.get_parameters()
                    },
                    "required": [p.name for p in self.get_parameters() if p.required],
                },
            },
        }
    
    def __repr__(self) -> str:
        return f"Tool(name={self.name}, description={self.description[:50]}...)"


class FunctionTool(Tool):
    """
    A tool created from a regular Python function.
    
    Wraps a function and automatically extracts its signature and documentation.
    """
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ):
        super().__init__(name=name, description=description, tags=tags)
        self.func = func
        
        # Extract parameters from function signature
        self._parameters = _extract_parameters(func)
        
        # Use function name if not provided
        if name is None:
            self._name = func.__name__
        
        # Use function docstring if description not provided
        if description is None:
            self._description = inspect.getdoc(func) or ""
    
    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped function."""
        result = self.func(**kwargs)
        if inspect.iscoroutine(result):
            return await result
        return result
    
    def get_parameters(self) -> list[ToolParameter]:
        """Get the parameters for this tool."""
        return self._parameters
    
    def __call__(self, **kwargs) -> Any:
        """Allow calling the tool directly."""
        return self.func(**kwargs)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator to convert a function into a tool.
    
    Args:
        name: Optional name for the tool (defaults to function name)
        description: Optional description (defaults to docstring)
        tags: Optional list of tags for categorization
    
    Returns:
        Decorator function that wraps the target function
    
    Example:
        @tool()
        def add(a: int, b: int) -> int:
            '''Add two numbers together.'''
            return a + b
    """
    def decorator(func: Callable) -> FunctionTool:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return FunctionTool(
            func=wrapper,
            name=name,
            description=description,
            tags=tags,
        )
    
    return decorator


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Provides methods for registering, retrieving, and listing tools.
    """
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_all(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_by_tag(self, tag: str) -> list[Tool]:
        """Get all tools with a specific tag."""
        return [t for t in self._tools.values() if tag in t.tags]
    
    def get_schema(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for all tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary format."""
        return {
            "tools": {
                name: tool.to_dict()
                for name, tool in self._tools.items()
            }
        }
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"
