"""
Base memory classes for TorchAgentic.

This module provides the foundation for memory systems that agents can use
to store and retrieve information.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid
import time
from enum import Enum


class MemoryType(str, Enum):
    """Types of memory storage."""
    
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    """
    Represents a single memory entry.
    
    Attributes:
        id: Unique identifier for the memory
        content: The actual memory content
        memory_type: Type of memory
        metadata: Additional metadata
        created_at: Timestamp when memory was created
        updated_at: Timestamp when memory was last updated
        access_count: Number of times this memory has been accessed
        importance: Importance score (0-1)
        expires_at: Optional expiration timestamp
    """
    
    content: Any
    memory_type: MemoryType = MemoryType.SHORT_TERM
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5
    expires_at: Optional[float] = None
    
    def __post_init__(self):
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)
    
    def touch(self) -> None:
        """Update the access timestamp and count."""
        self.access_count += 1
        self.updated_at = time.time()
    
    def is_expired(self) -> bool:
        """Check if the memory has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "importance": self.importance,
            "expires_at": self.expires_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content"),
            memory_type=MemoryType(data.get("memory_type", "short_term")),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            expires_at=data.get("expires_at"),
        )


class BaseMemory(ABC):
    """
    Abstract base class for memory systems.
    
    Defines the interface that all memory implementations must follow.
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self.capacity = capacity
        self._entries: dict[str, MemoryEntry] = {}
    
    @abstractmethod
    async def store(self, content: Any, **kwargs) -> str:
        """
        Store content in memory.
        
        Args:
            content: Content to store
            **kwargs: Additional arguments
            
        Returns:
            ID of the stored memory
        """
        pass
    
    @abstractmethod
    async def retrieve(self, query: Any, **kwargs) -> Optional[Any]:
        """
        Retrieve content from memory.
        
        Args:
            query: Query to search for
            **kwargs: Additional arguments
            
        Returns:
            Retrieved content or None
        """
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            id: ID of memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """
        Search for memories matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching memory entries
        """
        pass
    
    def get(self, id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        entry = self._entries.get(id)
        if entry and not entry.is_expired():
            entry.touch()
            return entry
        elif entry and entry.is_expired():
            self._entries.pop(id, None)
        return None
    
    def list_memories(self) -> list[MemoryEntry]:
        """List all non-expired memories."""
        now = time.time()
        return [
            entry for entry in self._entries.values()
            if not entry.is_expired()
        ]
    
    def count(self) -> int:
        """Get the count of non-expired memories."""
        return len(self.list_memories())
    
    def _enforce_capacity(self) -> None:
        """Enforce capacity limits by removing oldest/least important memories."""
        if self.capacity is None or len(self._entries) <= self.capacity:
            return
        
        # Sort by importance and recency
        entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.updated_at),
            reverse=True,
        )
        
        # Remove least important/oldest
        to_remove = len(self._entries) - self.capacity
        for entry in entries[-to_remove:]:
            self._entries.pop(entry.id, None)
    
    def __len__(self) -> int:
        return self.count()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(count={self.count()}, capacity={self.capacity})"


class Memory(BaseMemory):
    """
    Default memory implementation that combines short-term and long-term storage.
    
    This is a convenience class that provides a unified interface.
    """
    
    def __init__(
        self,
        short_term_capacity: int = 100,
        long_term_capacity: int = 1000,
        auto_promote: bool = True,
        promotion_threshold: int = 5,
    ):
        super().__init__()
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(capacity=long_term_capacity)
        self.auto_promote = auto_promote
        self.promotion_threshold = promotion_threshold
    
    async def store(
        self,
        content: Any,
        memory_type: Optional[MemoryType] = None,
        importance: float = 0.5,
        **kwargs,
    ) -> str:
        """Store content in appropriate memory based on type/importance."""
        mem_type = memory_type or MemoryType.SHORT_TERM
        
        entry = MemoryEntry(
            content=content,
            memory_type=mem_type,
            importance=importance,
            **kwargs,
        )
        
        if mem_type == MemoryType.LONG_TERM or importance > 0.8:
            entry_id = await self.long_term.store_entry(entry)
        else:
            entry_id = await self.short_term.store_entry(entry)
            
            # Auto-promote if accessed frequently
            if self.auto_promote and entry.access_count >= self.promotion_threshold:
                await self.long_term.store_entry(entry)
        
        return entry_id
    
    async def retrieve(self, query: Any, **kwargs) -> Optional[Any]:
        """Retrieve from both short and long-term memory."""
        # Try short-term first
        result = await self.short_term.retrieve(query, **kwargs)
        if result is not None:
            return result
        
        # Fall back to long-term
        return await self.long_term.retrieve(query, **kwargs)
    
    async def delete(self, id: str) -> bool:
        """Delete from either memory."""
        if await self.short_term.delete(id):
            return True
        return await self.long_term.delete(id)
    
    async def clear(self) -> None:
        """Clear both memories."""
        await self.short_term.clear()
        await self.long_term.clear()
    
    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search both memories."""
        st_results = await self.short_term.search(query, limit=limit)
        lt_results = await self.long_term.search(query, limit=limit)
        
        # Combine and sort by relevance/recency
        all_results = st_results + lt_results
        all_results.sort(key=lambda e: e.updated_at, reverse=True)
        
        return all_results[:limit]
    
    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "short_term": {
                "count": len(self.short_term),
                "capacity": self.short_term.capacity,
            },
            "long_term": {
                "count": len(self.long_term),
                "capacity": self.long_term.capacity,
            },
            "total": len(self),
        }
