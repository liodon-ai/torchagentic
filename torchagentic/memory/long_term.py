"""
Long-term memory implementation for TorchAgentic.

Provides persistent storage with semantic search capabilities.
"""

from typing import Any, Optional
import json
import os
import time
import hashlib

from torchagentic.memory.base import BaseMemory, MemoryEntry, MemoryType


class LongTermMemory(BaseMemory):
    """
    Long-term memory with optional vector search capabilities.
    
    Designed for persistent storage of important information.
    Supports file-based persistence and semantic search when
    embedding models are available.
    
    Attributes:
        capacity: Maximum number of entries to store
        storage_path: Optional path for file-based persistence
        use_embeddings: Whether to use embeddings for search
        embedding_model: Optional embedding model for semantic search
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        storage_path: Optional[str] = None,
        use_embeddings: bool = False,
        embedding_model: Optional[Any] = None,
    ):
        super().__init__(capacity=capacity)
        self.storage_path = storage_path
        self.use_embeddings = use_embeddings and embedding_model is not None
        self.embedding_model = embedding_model
        self._entries: dict[str, MemoryEntry] = {}
        self._index: dict[str, list[str]] = {}  # keyword -> entry_ids
        
        # Load from storage if path provided
        if storage_path:
            self._load_from_disk()
    
    async def store(
        self,
        content: Any,
        importance: float = 0.7,
        tags: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """
        Store content in long-term memory.
        
        Args:
            content: Content to store
            importance: Importance score (0-1), higher for long-term
            tags: Optional tags for categorization
            **kwargs: Additional arguments
            
        Returns:
            ID of the stored memory
        """
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.LONG_TERM,
            importance=importance,
            metadata={"tags": tags or []},
            **kwargs,
        )
        
        return await self.store_entry(entry)
    
    async def store_entry(self, entry: MemoryEntry) -> str:
        """Store a memory entry directly."""
        self._enforce_capacity()
        
        self._entries[entry.id] = entry
        
        # Update keyword index
        self._index_entry(entry)
        
        # Save to disk if configured
        if self.storage_path:
            self._save_to_disk()
        
        return entry.id
    
    async def retrieve(self, query: Any, **kwargs) -> Optional[Any]:
        """
        Retrieve content by ID or search query.
        """
        # If query is an ID, direct lookup
        if isinstance(query, str) and query in self._entries:
            entry = self._entries[query]
            if not entry.is_expired():
                entry.touch()
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
            entry = self._entries[id]
            del self._entries[id]
            self._remove_from_index(entry)
            
            if self.storage_path:
                self._save_to_disk()
            
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()
        self._index.clear()
        
        if self.storage_path:
            self._save_to_disk()
    
    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """
        Search memories using keyword matching or semantic search.
        """
        if self.use_embeddings and self.embedding_model is not None:
            return await self._semantic_search(query, limit)
        else:
            return await self._keyword_search(query, limit)
    
    async def _keyword_search(self, query: str, limit: int) -> list[MemoryEntry]:
        """Search using keyword matching."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores: dict[str, float] = {}
        
        # Check indexed keywords
        for word in query_words:
            if word in self._index:
                for entry_id in self._index[word]:
                    scores[entry_id] = scores.get(entry_id, 0) + 1
        
        # Also do full-text search for phrases
        for entry in self._entries.values():
            if entry.is_expired():
                continue
            
            content_str = str(entry.content).lower()
            
            # Exact phrase match gets high score
            if query_lower in content_str:
                scores[entry.id] = scores.get(entry.id, 0) + 5
            
            # Tag match
            tags = entry.metadata.get("tags", [])
            for tag in tags:
                if query_lower in tag.lower():
                    scores[entry.id] = scores.get(entry.id, 0) + 3
        
        # Calculate final scores with importance and recency
        now = time.time()
        results = []
        for entry_id, score in scores.items():
            entry = self._entries.get(entry_id)
            if entry and not entry.is_expired():
                # Boost by importance
                score *= (0.5 + entry.importance)
                
                # Slight recency boost
                age_days = (now - entry.created_at) / 86400
                recency_factor = 1 / (1 + age_days * 0.1)
                score *= recency_factor
                
                entry.metadata["search_score"] = score
                results.append(entry)
        
        # Sort by score
        results.sort(key=lambda e: e.metadata.get("search_score", 0), reverse=True)
        return results[:limit]
    
    async def _semantic_search(self, query: str, limit: int) -> list[MemoryEntry]:
        """Search using semantic embeddings (if available)."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Compare with stored embeddings
            scores = []
            for entry in self._entries.values():
                if entry.is_expired():
                    continue
                
                if "embedding" in entry.metadata:
                    import numpy as np
                    entry_embedding = np.array(entry.metadata["embedding"])
                    
                    # Cosine similarity
                    similarity = np.dot(query_embedding, entry_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entry_embedding)
                    )
                    
                    # Combine with importance
                    final_score = similarity * 0.7 + entry.importance * 0.3
                    entry.metadata["search_score"] = final_score
                    scores.append(entry)
            
            scores.sort(key=lambda e: e.metadata.get("search_score", 0), reverse=True)
            return scores[:limit]
        
        except Exception:
            # Fall back to keyword search on error
            return await self._keyword_search(query, limit)
    
    def _index_entry(self, entry: MemoryEntry) -> None:
        """Add entry to keyword index."""
        content_str = str(entry.content).lower()
        words = set(content_str.split())
        
        # Add tags to index
        tags = entry.metadata.get("tags", [])
        for tag in tags:
            words.add(tag.lower())
        
        # Index each word
        for word in words:
            # Skip very short words
            if len(word) < 3:
                continue
            
            if word not in self._index:
                self._index[word] = []
            
            if entry.id not in self._index[word]:
                self._index[word].append(entry.id)
        
        # Store embedding if enabled
        if self.use_embeddings and self.embedding_model is not None:
            try:
                import numpy as np
                embedding = self.embedding_model.encode([content_str])[0]
                entry.metadata["embedding"] = embedding.tolist()
            except Exception:
                pass
    
    def _remove_from_index(self, entry: MemoryEntry) -> None:
        """Remove entry from keyword index."""
        content_str = str(entry.content).lower()
        words = set(content_str.split())
        tags = entry.metadata.get("tags", [])
        for tag in tags:
            words.add(tag.lower())
        
        for word in words:
            if word in self._index:
                self._index[word] = [
                    eid for eid in self._index[word]
                    if eid != entry.id
                ]
                if not self._index[word]:
                    del self._index[word]
    
    def _enforce_capacity(self) -> None:
        """Enforce capacity by removing least important memories."""
        if self.capacity is None or len(self._entries) < self.capacity:
            return
        
        # Cleanup expired first
        expired = [
            eid for eid, entry in self._entries.items()
            if entry.is_expired()
        ]
        for eid in expired:
            self._entries.pop(eid, None)
        
        # If still over capacity, remove least important
        while len(self._entries) > self.capacity:
            # Find least important non-expired entry
            min_importance = float("inf")
            min_id = None
            
            for entry_id, entry in self._entries.items():
                if entry.importance < min_importance:
                    min_importance = entry.importance
                    min_id = entry_id
            
            if min_id:
                entry = self._entries.pop(min_id)
                self._remove_from_index(entry)
    
    def _save_to_disk(self) -> None:
        """Save memories to disk."""
        if not self.storage_path:
            return
        
        try:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            
            data = {
                "entries": [entry.to_dict() for entry in self._entries.values()],
                "index": self._index,
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            print(f"Warning: Failed to save long-term memory: {e}")
    
    def _load_from_disk(self) -> None:
        """Load memories from disk."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            for entry_data in data.get("entries", []):
                entry = MemoryEntry.from_dict(entry_data)
                if not entry.is_expired():
                    self._entries[entry.id] = entry
            
            self._index = data.get("index", {})
        
        except Exception as e:
            print(f"Warning: Failed to load long-term memory: {e}")
    
    def get_by_tag(self, tag: str) -> list[MemoryEntry]:
        """Get all memories with a specific tag."""
        return [
            entry for entry in self._entries.values()
            if not entry.is_expired() and tag in entry.metadata.get("tags", [])
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
            "type": "long_term",
            "capacity": self.capacity,
            "storage_path": self.storage_path,
            "use_embeddings": self.use_embeddings,
            "count": len(self),
            "entries": [entry.to_dict() for entry in self.list_memories()],
        }
    
    def __repr__(self) -> str:
        return (
            f"LongTermMemory(count={self.count()}, "
            f"capacity={self.capacity}, persistent={bool(self.storage_path)})"
        )
