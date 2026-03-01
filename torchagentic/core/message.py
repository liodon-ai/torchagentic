"""
Message handling for agent conversations.

This module provides classes for representing messages, conversation history,
and message roles in agent interactions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import uuid
import time


class MessageRole(str, Enum):
    """Enum representing the role of a message sender."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool/function call made by the agent."""
    
    id: str
    name: str
    arguments: dict[str, Any]
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    
    tool_call_id: str
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class Message:
    """
    Represents a single message in an agent conversation.
    
    Attributes:
        role: The role of the message sender (system, user, assistant, tool)
        content: The text content of the message
        tool_calls: List of tool calls if the message is from assistant
        tool_call_id: ID of the tool call this message responds to
        metadata: Additional metadata for the message
        timestamp: When the message was created
        id: Unique identifier for the message
    """
    
    role: MessageRole
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        result = {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "id": self.id,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        tool_calls = []
        if "tool_calls" in data:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                )
                for tc in data["tool_calls"]
            ]
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
            id=data.get("id", str(uuid.uuid4())),
        )
    
    @classmethod
    def system(cls, content: str, **kwargs) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)
    
    @classmethod
    def user(cls, content: str, **kwargs) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def assistant(cls, content: Optional[str] = None, tool_calls: Optional[list[ToolCall]] = None, **kwargs) -> "Message":
        """Create an assistant message."""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
            **kwargs,
        )
    
    @classmethod
    def tool(cls, content: Any, tool_call_id: str, **kwargs) -> "Message":
        """Create a tool response message."""
        return cls(
            role=MessageRole.TOOL,
            content=str(content),
            tool_call_id=tool_call_id,
            **kwargs,
        )


class Conversation:
    """
    Manages a conversation history between user and agent.
    
    Provides methods for adding messages, retrieving history,
    and managing conversation state.
    
    Attributes:
        messages: List of messages in the conversation
        system_prompt: Optional system prompt for the conversation
        max_messages: Maximum number of messages to retain
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_messages: Optional[int] = None,
    ):
        self.messages: list[Message] = []
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.id: str = str(uuid.uuid4())
        self.created_at: float = time.time()
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self._trim_if_needed()
    
    def add_system_message(self, content: str, **kwargs) -> Message:
        """Add a system message and return it."""
        message = Message.system(content, **kwargs)
        self.add_message(message)
        return message
    
    def add_user_message(self, content: str, **kwargs) -> Message:
        """Add a user message and return it."""
        message = Message.user(content, **kwargs)
        self.add_message(message)
        return message
    
    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        **kwargs,
    ) -> Message:
        """Add an assistant message and return it."""
        message = Message.assistant(content=content, tool_calls=tool_calls, **kwargs)
        self.add_message(message)
        return message
    
    def add_tool_result(
        self,
        result: Any,
        tool_call_id: str,
        **kwargs,
    ) -> Message:
        """Add a tool result message and return it."""
        message = Message.tool(result, tool_call_id=tool_call_id, **kwargs)
        self.add_message(message)
        return message
    
    def _trim_if_needed(self) -> None:
        """Trim messages if max_messages is set."""
        if self.max_messages and len(self.messages) > self.max_messages:
            # Keep system message if present, then keep most recent messages
            system_msg = None
            if self.messages and self.messages[0].role == MessageRole.SYSTEM:
                system_msg = self.messages[0]
                self.messages = self.messages[1:]
            
            # Remove oldest messages
            self.messages = self.messages[-(self.max_messages - 1):]
            
            # Re-add system message at the beginning
            if system_msg:
                self.messages.insert(0, system_msg)
    
    def get_messages(self) -> list[Message]:
        """Get all messages in the conversation."""
        return self.messages.copy()
    
    def get_messages_for_llm(self) -> list[dict[str, Any]]:
        """Get messages formatted for LLM API consumption."""
        messages = []
        
        # Add system prompt first if present
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        
        # Add all messages
        for msg in self.messages:
            msg_dict = msg.to_dict()
            # Remove internal fields
            msg_dict.pop("metadata", None)
            msg_dict.pop("timestamp", None)
            msg_dict.pop("id", None)
            messages.append(msg_dict)
        
        return messages
    
    def get_last_message(self, role: Optional[MessageRole] = None) -> Optional[Message]:
        """Get the last message, optionally filtered by role."""
        if role is None:
            return self.messages[-1] if self.messages else None
        
        for msg in reversed(self.messages):
            if msg.role == role:
                return msg
        return None
    
    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages = []
    
    def to_dict(self) -> dict[str, Any]:
        """Convert conversation to dictionary format."""
        return {
            "id": self.id,
            "system_prompt": self.system_prompt,
            "max_messages": self.max_messages,
            "created_at": self.created_at,
            "messages": [msg.to_dict() for msg in self.messages],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create a Conversation from a dictionary."""
        conv = cls(
            system_prompt=data.get("system_prompt"),
            max_messages=data.get("max_messages"),
        )
        conv.id = data.get("id", str(uuid.uuid4()))
        conv.created_at = data.get("created_at", time.time())
        conv.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return conv
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __repr__(self) -> str:
        return f"Conversation(id={self.id}, messages={len(self.messages)})"
