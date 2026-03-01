"""
Short-term memory implementation for TorchAgentic.

Provides fast, limited-capacity memory for recent interactions.
"""

from typing import Any, Optional
from collections import OrderedDict
import time

from torchagentic.memory.base import BaseMemory, MemoryEntry, MemoryType


class ShortTermMemory(BaseMemory):
    """
    Short-term memory with LRU eviction policy.
    
    Designed for storing recent conversation history and immediate context.
    Uses a combination of recency and importance for eviction decisions.
    
    Attributes:
        capacity: Maximum number of entries to store
        ttl: Optional time-to-live in seconds for entries
        decay_factor: Factor by which importance decays over time
    """
    
    def __init__(
        self,
        capacity: int = 100,
        ttl: Optional[float] = 3600,  # 1 hour default
        decay_factor: float = 0.001,
    ):
        super().__init__(capacity=capacity)
        self.ttl = ttl
        self.decay_factor = decay_factor
        self._order: OrderedDict[str, float] = OrderedDict()  # id -> last_access
        self._entries: dict[str, MemoryEntry] = {}
    
    async def store(
        self,
        content: Any,
        importance: float = 0.5,
        ttl: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Store content in short-term memory.
        
        Args:
            content: Content to store
            importance: Importance score (0-1)
            ttl: Optional TTL override
            **kwargs: Additional arguments
            
        Returns:
            ID of the stored memory
        """
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            importance=importance,
            expires_at=time.time() + (ttl or self.ttl) if (ttl or self.ttl) else None,
            **kwargs,
        )
        
        return await self.store_entry(entry)
    
    async def store_entry(self, entry: MemoryEntry) -> str:
        """Store a memory entry directly."""
        # Enforce capacity before adding
        self._enforce_capacity()
        
        self._entries[entry.id] = entry
        self._order[entry.id] = time.time()
        
        return entry.id
    
    async def retrieve(self, query: Any, **kwargs) -> Optional[Any]:
        """
        Retrieve content by ID or exact match.
        
        For more complex queries, use search().
        """
        # If query is an ID, direct lookup
        if isinstance(query, str) and query in self._entries:
            entry = self._entries[query]
            if not entry.is_expired():
                entry.touch()
                self._order.move_to_end(query)
                return entry.content
            else:
                await self.delete(query)
                return None
        
        # Otherwise, search for match
        results = await self.search(str(query), limit=1)
        if results:
            return results[0].content
        
        return None
    
    async def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        if id in self._entries:
            del self._entries[id]
            self._order.pop(id, None)
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()
        self._order.clear()
    
    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """
        Search memories by content matching.
        
        Simple substring search - for semantic search, use LongTermMemory
        with embeddings.
        """
        query_lower = query.lower()
        matches = []
        
        for entry in self._entries.values():
            if entry.is_expired():
                continue
            
            # Check content match
            content_str = str(entry.content).lower()
            metadata_str = str(entry.metadata).lower()
            
            if query_lower in content_str or query_lower in metadata_str:
                # Calculate relevance score based on match position and importance
                score = entry.importance
                if query_lower in content_str:
                    score += 0.5
                if query_lower in metadata_str:
                    score += 0.3
                
                # Boost recent memories
                recency_bonus = (time.time() - entry.created_at) / 3600  # hours
                score += recency_bonus * self.decay_factor
                
                entry.metadata["search_score"] = score
                matches.append(entry)
        
        # Sort by score and return top results
        matches.sort(key=lambda e: e.metadata.get("search_score", 0), reverse=True)
        return matches[:limit]
    
    def _enforce_capacity(self) -> None:
        """Enforce capacity using LRU + importance-based eviction."""
        if self.capacity is None or len(self._entries) < self.capacity:
            return
        
        # Apply TTL expiration first
        self._cleanup_expired()
        
        # If still over capacity, evict based on combined score
        while len(self._entries) > self.capacity and self._order:
            # Find entry with lowest combined score (importance * recency)
            now = time.time()
            min_score = float("inf")
            min_id = None
            
            for entry_id in list(self._order.keys()):
                entry = self._entries.get(entry_id)
                if entry is None:
                    continue
                
                # Calculate eviction score
                recency = now - self._order[entry_id]
                score = entry.importance * (1 / (recency + 1))
                
                if score < min_score:
                    min_score = score
                    min_id = entry_id
            
            if min_id:
                del self._entries[min_id]
                self._order.pop(min_id, None)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired = [
            entry_id for entry_id, entry in self._entries.items()
            if entry.is_expired()
        ]
        for entry_id in expired:
            del self._entries[entry_id]
            self._order.pop(entry_id, None)
    
    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Get the most recently accessed memories."""
        recent_ids = list(self._order.keys())[-limit:]
        return [
            self._entries[entry_id]
            for entry_id in recent_ids
            if entry_id in self._entries and not self._entries[entry_id].is_expired()
        ]
    
    def get_by_importance(self, min_importance: float = 0.5) -> list[MemoryEntry]:
        """Get memories above a certain importance threshold."""
        return [
            entry for entry in self._entries.values()
            if not entry.is_expired() and entry.importance >= min_importance
        ]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "type": "short_term",
            "capacity": self.capacity,
            "ttl": self.ttl,
            "count": len(self),
            "entries": [entry.to_dict() for entry in self.list_memories()],
        }
    
    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(count={self.count()}, "
            f"capacity={self.capacity}, ttl={self.ttl})"
        )
