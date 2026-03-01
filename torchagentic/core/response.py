"""
Response handling for agent outputs.

This module provides classes for representing agent responses,
including status, content, and metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class ResponseStatus(str, Enum):
    """Enum representing the status of an agent response."""
    
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TokenUsage:
    """Tracks token usage for an LLM call."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class AgentResponse:
    """
    Represents the response from an agent execution.
    
    Attributes:
        content: The main text content of the response
        status: The status of the response
        conversation_id: ID of the conversation this response belongs to
        tool_calls: List of tool calls the agent wants to make
        tool_results: Results from executed tool calls
        metadata: Additional metadata about the response
        token_usage: Token usage statistics
        execution_time: Time taken to generate the response
        error: Error message if status is ERROR
        timestamp: When the response was generated
    """
    
    content: Optional[str] = None
    status: ResponseStatus = ResponseStatus.SUCCESS
    conversation_id: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[TokenUsage] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ResponseStatus(self.status)
    
    @property
    def is_success(self) -> bool:
        """Check if the response was successful."""
        return self.status == ResponseStatus.SUCCESS
    
    @property
    def is_error(self) -> bool:
        """Check if the response contains an error."""
        return self.status == ResponseStatus.ERROR
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary format."""
        result = {
            "content": self.content,
            "status": self.status.value,
            "conversation_id": self.conversation_id,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "error": self.error,
            "timestamp": self.timestamp,
        }
        if self.token_usage:
            result["token_usage"] = {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
            }
        return result
    
    @classmethod
    def success(
        cls,
        content: str,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> "AgentResponse":
        """Create a successful response."""
        return cls(
            content=content,
            status=ResponseStatus.SUCCESS,
            tool_calls=tool_calls or [],
            **kwargs,
        )
    
    @classmethod
    def error(cls, error: str, **kwargs) -> "AgentResponse":
        """Create an error response."""
        return cls(
            content=None,
            status=ResponseStatus.ERROR,
            error=error,
            **kwargs,
        )
    
    @classmethod
    def partial(
        cls,
        content: str,
        **kwargs,
    ) -> "AgentResponse":
        """Create a partial response (for streaming)."""
        return cls(
            content=content,
            status=ResponseStatus.PARTIAL,
            **kwargs,
        )
    
    def __repr__(self) -> str:
        status_str = self.status.value
        if self.error:
            return f"AgentResponse(status={status_str}, error={self.error})"
        content_preview = (self.content or "")[:50]
        return f"AgentResponse(status={status_str}, content='{content_preview}...')"
