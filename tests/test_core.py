"""
Tests for core module.
"""

import pytest
import asyncio
from torchagentic.core.message import Message, MessageRole, Conversation, ToolCall
from torchagentic.core.response import AgentResponse, ResponseStatus, TokenUsage
from torchagentic.core.agent import Agent
from torchagentic.llms.local import MockLLM


class TestMessage:
    """Tests for Message class."""
    
    def test_create_user_message(self):
        msg = Message.user("Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
    
    def test_create_system_message(self):
        msg = Message.system("You are helpful")
        assert msg.role == MessageRole.SYSTEM
    
    def test_create_assistant_message(self):
        msg = Message.assistant("Hi there")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there"
    
    def test_create_tool_message(self):
        msg = Message.tool("Result", tool_call_id="call_123")
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call_123"
    
    def test_message_to_dict(self):
        msg = Message.user("Test")
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Test"
    
    def test_message_from_dict(self):
        data = {"role": "user", "content": "Test"}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.USER
        assert msg.content == "Test"
    
    def test_tool_call(self):
        tc = ToolCall(id="1", name="test", arguments={"a": 1})
        assert tc.name == "test"
        assert tc.arguments == {"a": 1}


class TestConversation:
    """Tests for Conversation class."""
    
    def test_create_conversation(self):
        conv = Conversation(system_prompt="Be helpful")
        assert conv.system_prompt == "Be helpful"
        assert len(conv) == 0
    
    def test_add_messages(self):
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")
        assert len(conv) == 2
    
    def test_get_messages_for_llm(self):
        conv = Conversation(system_prompt="System")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")
        
        messages = conv.get_messages_for_llm()
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
    
    def test_max_messages(self):
        conv = Conversation(max_messages=3)
        for i in range(5):
            conv.add_user_message(f"Message {i}")
        assert len(conv) <= 3
    
    def test_clear_conversation(self):
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.clear()
        assert len(conv) == 0
    
    def test_conversation_to_dict(self):
        conv = Conversation(system_prompt="Test")
        data = conv.to_dict()
        assert data["system_prompt"] == "Test"
    
    def test_conversation_from_dict(self):
        data = {"system_prompt": "Test", "messages": []}
        conv = Conversation.from_dict(data)
        assert conv.system_prompt == "Test"


class TestAgentResponse:
    """Tests for AgentResponse class."""
    
    def test_success_response(self):
        resp = AgentResponse.success("Done")
        assert resp.is_success
        assert resp.content == "Done"
    
    def test_error_response(self):
        resp = AgentResponse.error("Something failed")
        assert resp.is_error
        assert resp.error == "Something failed"
    
    def test_partial_response(self):
        resp = AgentResponse.partial("Partial result")
        assert resp.status == ResponseStatus.PARTIAL
    
    def test_response_with_tool_calls(self):
        resp = AgentResponse.success(
            "Processing",
            tool_calls=[{"name": "test", "arguments": {}}],
        )
        assert resp.has_tool_calls
    
    def test_response_to_dict(self):
        resp = AgentResponse.success("Test")
        data = resp.to_dict()
        assert data["status"] == "success"
        assert data["content"] == "Test"


class TestTokenUsage:
    """Tests for TokenUsage class."""
    
    def test_token_usage(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150
    
    def test_token_usage_explicit(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=200)
        assert usage.total_tokens == 200


class TestAgent:
    """Tests for Agent class."""
    
    @pytest.mark.asyncio
    async def test_create_agent(self):
        llm = MockLLM()
        agent = Agent(llm=llm, name="TestAgent")
        assert agent.name == "TestAgent"
    
    @pytest.mark.asyncio
    async def test_agent_act(self):
        llm = MockLLM(response="Hello!")
        agent = Agent(llm=llm)
        response = await agent.act("Hi")
        assert response.content == "Hello!"
    
    @pytest.mark.asyncio
    async def test_agent_chat(self):
        llm = MockLLM(response="Sure!")
        agent = Agent(llm=llm)
        response = await agent.chat("Help me")
        assert response.is_success
    
    @pytest.mark.asyncio
    async def test_agent_state(self):
        llm = MockLLM()
        agent = Agent(llm=llm, temperature=0.5)
        state = agent.get_state()
        assert state["temperature"] == 0.5
    
    @pytest.mark.asyncio
    async def test_agent_set_state(self):
        llm = MockLLM()
        agent = Agent(llm=llm)
        agent.set_state({"temperature": 0.9})
        assert agent.temperature == 0.9
    
    @pytest.mark.asyncio
    async def test_agent_to_dict(self):
        llm = MockLLM()
        agent = Agent(llm=llm, name="DictTest")
        data = agent.to_dict()
        assert data["name"] == "DictTest"
        assert data["type"] == "Agent"
    
    @pytest.mark.asyncio
    async def test_agent_conversation(self):
        llm = MockLLM()
        agent = Agent(llm=llm)
        
        # First message creates conversation
        await agent.act("First message")
        conv_id = agent._current_conversation_id
        assert conv_id is not None
        
        # Second message uses same conversation
        await agent.act("Second message")
        conv = agent.get_conversation()
        assert len(conv) == 4  # 2 user + 2 assistant messages
    
    @pytest.mark.asyncio
    async def test_agent_new_conversation(self):
        llm = MockLLM()
        agent = Agent(llm=llm)
        
        await agent.act("First")
        old_conv_id = agent._current_conversation_id
        
        new_id = agent.new_conversation()
        assert new_id != old_conv_id
        assert agent._current_conversation_id == new_id
