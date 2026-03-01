"""
Memory Example

This example demonstrates how to use short-term and long-term memory
with agents.
"""

import asyncio
from torchagentic import Agent, ShortTermMemory, LongTermMemory, Memory
from torchagentic.llms.local import MockLLM


async def main():
    print("=" * 50)
    print("TorchAgentic - Memory Example")
    print("=" * 50)
    
    # Create LLM
    llm = MockLLM(response="I remember our conversation.")
    
    print("\n[1] Short-Term Memory Example")
    print("-" * 30)
    
    # Create short-term memory
    short_term = ShortTermMemory(
        capacity=5,  # Only keep 5 items
        ttl=3600,    # 1 hour TTL
    )
    
    # Store some memories
    print("Storing memories in short-term memory...")
    memories = [
        {"type": "fact", "content": "The user likes Python programming"},
        {"type": "preference", "content": "Prefers dark mode in IDEs"},
        {"type": "context", "content": "Working on a web scraping project"},
        {"type": "fact", "content": "Has 5 years of coding experience"},
        {"type": "goal", "content": "Want to learn machine learning"},
    ]
    
    for mem in memories:
        entry_id = await short_term.store(
            mem,
            importance=0.7,
        )
        print(f"  Stored: {mem['content'][:30]}... (ID: {entry_id[:8]})")
    
    # Search memories
    print("\nSearching for 'python'...")
    results = await short_term.search("python", limit=3)
    for result in results:
        print(f"  Found: {result.content}")
    
    # Get recent memories
    print("\nRecent memories:")
    recent = short_term.get_recent(limit=3)
    for mem in recent:
        print(f"  - {mem.content}")
    
    # Get important memories
    print("\nImportant memories (>0.6):")
    important = short_term.get_by_importance(0.6)
    for mem in important:
        print(f"  - {mem.content} (importance: {mem.importance})")
    
    print(f"\nShort-term memory stats: {short_term}")
    
    print("\n[2] Long-Term Memory Example")
    print("-" * 30)
    
    # Create long-term memory with persistence
    long_term = LongTermMemory(
        capacity=100,
        storage_path="examples/memory_storage.json",
    )
    
    # Store important memories
    print("Storing memories in long-term memory...")
    important_memories = [
        {"type": "user_profile", "content": "Software engineer from Seattle"},
        {"type": "skills", "content": "Python, JavaScript, SQL, Docker"},
        {"type": "project", "content": "Building an AI assistant framework"},
    ]
    
    for mem in important_memories:
        entry_id = await long_term.store(
            mem,
            importance=0.9,  # High importance for long-term
            tags=[mem["type"]],
        )
        print(f"  Stored: {mem['content'][:30]}... (ID: {entry_id[:8]})")
    
    # Search long-term memory
    print("\nSearching for 'engineer'...")
    results = await long_term.search("engineer", limit=5)
    for result in results:
        print(f"  Found: {result.content}")
    
    # Get by tag
    print("\nMemories tagged with 'skills':")
    skills = long_term.get_by_tag("skills")
    for mem in skills:
        print(f"  - {mem.content}")
    
    print(f"\nLong-term memory stats: {long_term}")
    
    print("\n[3] Combined Memory with Agent")
    print("-" * 30)
    
    # Create memory system
    memory = Memory(
        short_term_capacity=50,
        long_term_capacity=500,
        auto_promote=True,
        promotion_threshold=3,
    )
    
    # Create agent with memory
    agent = Agent(
        llm=llm,
        name="MemoryAgent",
        system_prompt="You are a helpful assistant with excellent memory.",
        memory=memory,
        verbose=True,
    )
    
    # Simulate conversation
    print("Simulating conversation...")
    conversations = [
        "My name is Alex and I'm a developer.",
        "I work on machine learning projects.",
        "I prefer working with PyTorch over TensorFlow.",
        "My favorite programming language is Python.",
    ]
    
    for msg in conversations:
        print(f"\nUser: {msg}")
        response = await agent.act(msg)
        print(f"Agent: {response.content}")
    
    # Show memory stats
    print("\nMemory statistics:")
    stats = memory.get_stats()
    print(f"  Short-term: {stats['short_term']['count']} / {stats['short_term']['capacity']}")
    print(f"  Long-term: {stats['long_term']['count']} / {stats['long_term']['capacity']}")
    print(f"  Total: {stats['total']}")
    
    print("\n[4] Memory Persistence")
    print("-" * 30)
    
    # Long-term memory is automatically saved to disk
    print(f"Long-term memory saved to: examples/memory_storage.json")
    
    # Load from disk
    new_long_term = LongTermMemory(
        capacity=100,
        storage_path="examples/memory_storage.json",
    )
    print(f"Loaded {len(new_long_term)} memories from disk")
    
    # Cleanup
    import os
    if os.path.exists("examples/memory_storage.json"):
        os.remove("examples/memory_storage.json")
        print("Cleaned up memory storage file")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
