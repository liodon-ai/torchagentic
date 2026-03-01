"""
Base agent implementation for TorchAgentic.

This module provides the core agent classes that form the foundation
of the agentic system.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import asyncio
import time
import uuid

import torch

from torchagentic.core.message import Conversation, Message, MessageRole, ToolCall
from torchagentic.core.response import AgentResponse, ResponseStatus, TokenUsage
from torchagentic.tools.manager import ToolManager
from torchagentic.memory.base import BaseMemory
from torchagentic.llms.base import BaseLLM


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Defines the interface that all agents must implement.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or f"agent_{uuid.uuid4().hex[:8]}"
        self.id: str = str(uuid.uuid4())
        self.created_at: float = time.time()
    
    @abstractmethod
    async def act(self, input: Any, **kwargs) -> AgentResponse:
        """
        Execute the agent with the given input.
        
        Args:
            input: The input to process
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse containing the result
        """
        pass
    
    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get the current state of the agent."""
        pass
    
    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Set the state of the agent."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, id={self.id})"


class Agent(BaseAgent):
    """
    Main agent implementation that uses an LLM for decision making.
    
    This agent can use tools, maintain memory, and manage conversations.
    
    Attributes:
        llm: The language model to use for generating responses
        tool_manager: Manager for available tools
        memory: Optional memory system for the agent
        system_prompt: System prompt for the agent
        max_iterations: Maximum iterations for tool execution loops
        temperature: Temperature for LLM sampling
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[list[Callable]] = None,
        memory: Optional[BaseMemory] = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        super().__init__(name=name)
        
        self.llm = llm
        self.tool_manager = ToolManager()
        if tools:
            for tool in tools:
                self.tool_manager.register(tool)
        
        self.memory = memory
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.verbose = verbose
        
        # Conversation tracking
        self._conversations: dict[str, Conversation] = {}
        self._current_conversation_id: Optional[str] = None
    
    def _get_or_create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """Get or create a conversation."""
        if conversation_id is None:
            if self._current_conversation_id is None:
                self._current_conversation_id = str(uuid.uuid4())
                self._conversations[self._current_conversation_id] = Conversation(
                    system_prompt=self.system_prompt
                )
            conversation_id = self._current_conversation_id
        
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = Conversation(
                system_prompt=self.system_prompt
            )
        
        return self._conversations[conversation_id]
    
    def get_conversation(self, conversation_id: Optional[str] = None) -> Optional[Conversation]:
        """Get a conversation by ID."""
        conv_id = conversation_id or self._current_conversation_id
        return self._conversations.get(conv_id)
    
    def new_conversation(self, system_prompt: Optional[str] = None) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid.uuid4())
        self._conversations[conv_id] = Conversation(
            system_prompt=system_prompt or self.system_prompt
        )
        self._current_conversation_id = conv_id
        return conv_id
    
    def register_tool(self, tool: Callable) -> None:
        """Register a tool with the agent."""
        self.tool_manager.register(tool)
    
    def register_tools(self, tools: list[Callable]) -> None:
        """Register multiple tools with the agent."""
        for tool in tools:
            self.tool_manager.register(tool)
    
    async def _execute_tool_loop(
        self,
        conversation: Conversation,
    ) -> AgentResponse:
        """
        Execute the tool calling loop.
        
        The agent will iteratively:
        1. Generate a response (potentially with tool calls)
        2. Execute any tool calls
        3. Feed results back to the LLM
        4. Repeat until no more tool calls or max iterations reached
        """
        iteration = 0
        last_response: Optional[AgentResponse] = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Generate response from LLM
            response = await self._generate_response(conversation)
            last_response = response
            
            if response.is_error:
                return response
            
            # If no tool calls, we're done
            if not response.has_tool_calls:
                return response
            
            # Execute tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                result = await self.tool_manager.execute(
                    tool_call["name"],
                    tool_call.get("arguments", {}),
                )
                tool_results.append({
                    "tool_call_id": tool_call.get("id"),
                    "result": result.get("result"),
                    "error": result.get("error"),
                })
                
                # Add tool result to conversation
                conversation.add_tool_result(
                    result.get("result", result.get("error")),
                    tool_call_id=tool_call.get("id", ""),
                )
            
            # Update response with tool results
            response.tool_results = tool_results
            
            if self.verbose:
                print(f"[{self.name}] Iteration {iteration}: Executed {len(tool_results)} tool(s)")
        
        # Max iterations reached
        if last_response:
            last_response.status = ResponseStatus.PARTIAL
            last_response.metadata["max_iterations_reached"] = True
        
        return last_response or AgentResponse.error("No response generated")
    
    async def _generate_response(self, conversation: Conversation) -> AgentResponse:
        """Generate a response from the LLM."""
        start_time = time.time()
        
        # Get available tools schema
        tools_schema = self.tool_manager.get_schema()
        
        # Generate response
        response = await self.llm.generate(
            messages=conversation.get_messages_for_llm(),
            tools=tools_schema if tools_schema else None,
            temperature=self.temperature,
        )
        
        execution_time = time.time() - start_time
        
        # Parse the response
        content = response.get("content")
        tool_calls_raw = response.get("tool_calls", [])
        token_usage = response.get("token_usage")
        
        # Create ToolCall objects
        tool_calls = []
        for tc in tool_calls_raw:
            tool_calls.append({
                "id": tc.get("id", str(uuid.uuid4())),
                "name": tc.get("name", ""),
                "arguments": tc.get("arguments", tc.get("parameters", {})),
            })
        
        # Add assistant message to conversation
        conversation.add_assistant_message(
            content=content,
            tool_calls=[
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in tool_calls
            ] if tool_calls else None,
        )
        
        # Build response
        agent_response = AgentResponse(
            content=content,
            conversation_id=conversation.id,
            tool_calls=tool_calls,
            token_usage=TokenUsage(**token_usage) if token_usage else None,
            execution_time=execution_time,
        )
        
        return agent_response
    
    async def act(
        self,
        input: str,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Execute the agent with the given input.
        
        Args:
            input: User input to process
            conversation_id: Optional conversation ID to use
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse containing the result
        """
        # Get or create conversation
        conversation = self._get_or_create_conversation(conversation_id)
        
        # Add user message
        conversation.add_user_message(input)
        
        # Store in memory if available
        if self.memory:
            await self.memory.store({
                "type": "user_input",
                "content": input,
                "conversation_id": conversation.id,
                "timestamp": time.time(),
            })
        
        # Execute tool loop
        response = await self._execute_tool_loop(conversation)
        
        # Store response in memory if available
        if self.memory and response.content:
            await self.memory.store({
                "type": "assistant_output",
                "content": response.content,
                "conversation_id": conversation.id,
                "timestamp": time.time(),
            })
        
        if self.verbose:
            print(f"[{self.name}] Response: {response.content}")
        
        return response
    
    async def chat(
        self,
        message: str,
        **kwargs,
    ) -> AgentResponse:
        """
        Chat with the agent (alias for act with clearer semantics).
        
        Args:
            message: Message to send
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse containing the result
        """
        return await self.act(message, **kwargs)
    
    def get_state(self) -> dict[str, Any]:
        """Get the current state of the agent."""
        return {
            "name": self.name,
            "id": self.id,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "conversation_ids": list(self._conversations.keys()),
            "current_conversation_id": self._current_conversation_id,
            "tool_count": len(self.tool_manager.tools),
        }
    
    def set_state(self, state: dict[str, Any]) -> None:
        """Set the state of the agent."""
        if "system_prompt" in state:
            self.system_prompt = state["system_prompt"]
        if "temperature" in state:
            self.temperature = state["temperature"]
        if "max_iterations" in state:
            self.max_iterations = state["max_iterations"]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert agent to dictionary format."""
        return {
            "name": self.name,
            "id": self.id,
            "type": self.__class__.__name__,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "tools": self.tool_manager.get_schema(),
            "conversations": {
                cid: conv.to_dict()
                for cid, conv in self._conversations.items()
            },
        }
    
    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name}, "
            f"llm={self.llm.__class__.__name__}, "
            f"tools={len(self.tool_manager.tools)})"
        )
