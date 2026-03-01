"""
Base LLM classes for TorchAgentic.

This module provides the foundation for LLM integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time


@dataclass
class LLMConfig:
    """
    Configuration for LLM instances.
    
    Attributes:
        model: Model name/identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty for repetition
        presence_penalty: Presence penalty for repetition
        timeout: Request timeout in seconds
        api_key: Optional API key
        api_base: Optional API base URL
        extra: Additional provider-specific settings
    """
    
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 60.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.timeout,
            "api_key": self.api_key[:8] + "..." if self.api_key else None,
            "api_base": self.api_base,
            "extra": self.extra,
        }
    
    def copy(self, **overrides) -> "LLMConfig":
        """Create a copy with overrides."""
        return LLMConfig(
            model=overrides.get("model", self.model),
            temperature=overrides.get("temperature", self.temperature),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            top_p=overrides.get("top_p", self.top_p),
            frequency_penalty=overrides.get("frequency_penalty", self.frequency_penalty),
            presence_penalty=overrides.get("presence_penalty", self.presence_penalty),
            timeout=overrides.get("timeout", self.timeout),
            api_key=overrides.get("api_key", self.api_key),
            api_base=overrides.get("api_base", self.api_base),
            extra=overrides.get("extra", {**self.extra}),
        )


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    Defines the interface that all LLM implementations must follow.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "errors": 0,
        }
    
    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool schemas for function calling
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Dict with 'content', 'tool_calls', and 'token_usage'
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ):
        """
        Generate a streaming response.
        
        Yields:
            Chunks of the response
        """
        pass
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def _update_stats(
        self,
        tokens: int = 0,
        time_taken: float = 0.0,
        error: bool = False,
    ) -> None:
        """Update internal statistics."""
        self._stats["total_requests"] += 1
        self._stats["total_tokens"] += tokens
        self._stats["total_time"] += time_taken
        if error:
            self._stats["errors"] += 1
    
    def get_stats(self) -> dict[str, Any]:
        """Get LLM statistics."""
        stats = self._stats.copy()
        stats["avg_time"] = (
            stats["total_time"] / max(1, stats["total_requests"])
        )
        stats["avg_tokens"] = (
            stats["total_tokens"] / max(1, stats["total_requests"])
        )
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "errors": 0,
        }
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"
