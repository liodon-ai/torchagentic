"""
Basic Agent Example

This example demonstrates how to create and use a simple AI agent
with TorchAgentic.
"""

import asyncio
from torchagentic import Agent
from torchagentic.llms.local import LocalLLM, MockLLM


async def main():
    print("=" * 50)
    print("TorchAgentic - Basic Agent Example")
    print("=" * 50)
    
    # Option 1: Use MockLLM for testing (no model download required)
    print("\n[1] Using MockLLM for demonstration...")
    mock_llm = MockLLM(response="Hello! I'm a mock agent. I can help you with various tasks.")
    
    mock_agent = Agent(
        llm=mock_llm,
        name="MockAgent",
        system_prompt="You are a helpful assistant.",
        verbose=True,
    )
    
    response = await mock_agent.act("Hello!")
    print(f"Response: {response.content}")
    
    # Option 2: Use LocalLLM with a real model (requires transformers)
    print("\n[2] To use a real local model, uncomment the code below:")
    print("""
    from torchagentic.llms.local import LocalLLM
    
    llm = LocalLLM(
        model_id="microsoft/phi-2",  # or any HuggingFace model
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    agent = Agent(
        llm=llm,
        name="LocalAgent",
        system_prompt="You are a helpful AI assistant.",
        temperature=0.7,
        max_iterations=5,
        verbose=True,
    )
    
    response = await agent.act("What is machine learning?")
    print(f"Response: {response.content}")
    """)
    
    # Demonstrate conversation
    print("\n[3] Multi-turn conversation example:")
    mock_llm2 = MockLLM(response="I understand your question. Let me think about that...")
    
    agent2 = Agent(
        llm=mock_llm2,
        name="ConversationalAgent",
        system_prompt="You are a friendly conversational assistant.",
    )
    
    messages = [
        "What is your name?",
        "Can you help me with coding?",
        "Thank you for your help!",
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = await agent2.act(msg)
        print(f"Agent: {response.content}")
    
    # Show agent state
    print("\n[4] Agent state:")
    state = agent2.get_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
