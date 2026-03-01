"""
Tests for memory module.
"""

import pytest
import asyncio
from torchagentic.memory.base import Memory, MemoryEntry, MemoryType
from torchagentic.memory.short_term import ShortTermMemory
from torchagentic.memory.long_term import LongTermMemory


class TestMemoryEntry:
    """Tests for MemoryEntry class."""
    
    def test_create_entry(self):
        entry = MemoryEntry(content="Test content")
        assert entry.content == "Test content"
        assert entry.memory_type == MemoryType.SHORT_TERM
    
    def test_entry_to_dict(self):
        entry = MemoryEntry(content="Test")
        data = entry.to_dict()
        assert data["content"] == "Test"
    
    def test_entry_from_dict(self):
        data = {"content": "Test", "memory_type": "long_term"}
        entry = MemoryEntry.from_dict(data)
        assert entry.content == "Test"
        assert entry.memory_type == MemoryType.LONG_TERM
    
    def test_entry_touch(self):
        entry = MemoryEntry(content="Test")
        initial_count = entry.access_count
        entry.touch()
        assert entry.access_count == initial_count + 1
    
    def test_entry_expiration(self):
        import time
        entry = MemoryEntry(
            content="Test",
            expires_at=time.time() - 1,  # Expired 1 second ago
        )
        assert entry.is_expired()
        
        entry2 = MemoryEntry(
            content="Test",
            expires_at=time.time() + 3600,  # Expires in 1 hour
        )
        assert not entry2.is_expired()


class TestShortTermMemory:
    """Tests for ShortTermMemory class."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        memory = ShortTermMemory(capacity=10)
        
        entry_id = await memory.store("Test content")
        result = await memory.retrieve(entry_id)
        assert result == "Test content"
    
    @pytest.mark.asyncio
    async def test_search(self):
        memory = ShortTermMemory(capacity=10)
        
        await memory.store("Python is great")
        await memory.store("Java is also good")
        await memory.store("I love coding")
        
        results = await memory.search("python", limit=5)
        assert len(results) >= 1
        assert "Python" in results[0].content
    
    @pytest.mark.asyncio
    async def test_capacity_limit(self):
        memory = ShortTermMemory(capacity=3)
        
        for i in range(5):
            await memory.store(f"Content {i}")
        
        assert len(memory) <= 3
    
    @pytest.mark.asyncio
    async def test_delete(self):
        memory = ShortTermMemory(capacity=10)
        
        entry_id = await memory.store("To delete")
        result = await memory.delete(entry_id)
        assert result is True
        
        retrieved = await memory.retrieve(entry_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_clear(self):
        memory = ShortTermMemory(capacity=10)
        
        for i in range(5):
            await memory.store(f"Content {i}")
        
        await memory.clear()
        assert len(memory) == 0
    
    @pytest.mark.asyncio
    async def test_get_recent(self):
        memory = ShortTermMemory(capacity=10)
        
        for i in range(5):
            await memory.store(f"Content {i}")
        
        recent = memory.get_recent(limit=3)
        assert len(recent) == 3
    
    @pytest.mark.asyncio
    async def test_get_by_importance(self):
        memory = ShortTermMemory(capacity=10)
        
        await memory.store("Low importance", importance=0.3)
        await memory.store("High importance", importance=0.9)
        
        important = memory.get_by_importance(min_importance=0.5)
        assert len(important) == 1
        assert important[0].content == "High importance"


class TestLongTermMemory:
    """Tests for LongTermMemory class."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        memory = LongTermMemory(capacity=10)
        
        entry_id = await memory.store("Test content")
        result = await memory.retrieve(entry_id)
        assert result == "Test content"
    
    @pytest.mark.asyncio
    async def test_search(self):
        memory = LongTermMemory(capacity=10)
        
        await memory.store("Machine learning is fascinating")
        await memory.store("Deep learning uses neural networks")
        await memory.store("Python is popular for AI")
        
        results = await memory.search("learning", limit=5)
        assert len(results) >= 2
    
    @pytest.mark.asyncio
    async def test_tags(self):
        memory = LongTermMemory(capacity=10)
        
        await memory.store("Skill 1", tags=["skills"])
        await memory.store("Skill 2", tags=["skills"])
        await memory.store("Fact 1", tags=["facts"])
        
        skills = memory.get_by_tag("skills")
        assert len(skills) == 2
    
    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        storage_path = tmp_path / "memory.json"
        
        # Create and store
        memory1 = LongTermMemory(capacity=10, storage_path=str(storage_path))
        await memory1.store("Persistent content")
        
        # Load from disk
        memory2 = LongTermMemory(capacity=10, storage_path=str(storage_path))
        assert len(memory2) == 1
    
    @pytest.mark.asyncio
    async def test_capacity_limit(self):
        memory = LongTermMemory(capacity=3)
        
        for i in range(5):
            await memory.store(f"Content {i}", importance=0.5)
        
        assert len(memory) <= 3


class TestMemory:
    """Tests for combined Memory class."""
    
    @pytest.mark.asyncio
    async def test_combined_memory(self):
        memory = Memory(
            short_term_capacity=10,
            long_term_capacity=100,
        )
        
        # Store in short-term
        entry_id = await memory.store("Short-term content", importance=0.5)
        result = await memory.retrieve(entry_id)
        assert result == "Short-term content"
    
    @pytest.mark.asyncio
    async def test_auto_promote(self):
        memory = Memory(
            short_term_capacity=10,
            long_term_capacity=100,
            auto_promote=True,
            promotion_threshold=1,
        )
        
        # Store with high importance goes to long-term
        await memory.store("Important", importance=0.9)
        
        stats = memory.get_stats()
        assert stats["long_term"]["count"] >= 1
    
    @pytest.mark.asyncio
    async def test_search_both_memories(self):
        memory = Memory()
        
        await memory.store("Short-term item")
        await memory.store("Long-term item", importance=0.9)
        
        results = await memory.search("item", limit=10)
        assert len(results) >= 2
