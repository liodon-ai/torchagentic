"""
Configuration management for TorchAgentic.

Provides a flexible configuration system with support for
environment variables, files, and programmatic overrides.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Config:
    """
    Configuration manager for TorchAgentic.
    
    Supports loading from:
    - Default values
    - JSON/YAML config files
    - Environment variables
    - Programmatic overrides
    
    Attributes:
        llm_provider: Default LLM provider
        llm_model: Default LLM model
        api_key: API key for LLM provider
        max_tokens: Default max tokens
        temperature: Default temperature
        verbose: Enable verbose logging
        log_level: Logging level
        cache_dir: Directory for caching
    """
    
    # LLM Settings
    llm_provider: str = "local"
    llm_model: str = "microsoft/phi-2"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Generation Settings
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    
    # Agent Settings
    max_iterations: int = 10
    tool_timeout: float = 30.0
    
    # Memory Settings
    memory_capacity: int = 1000
    memory_persistence: bool = True
    
    # Logging Settings
    verbose: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Cache Settings
    cache_dir: Optional[str] = None
    
    # Extra settings
    extra: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set default cache dir
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "torchagentic")
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls, prefix: str = "TORCHAGENTIC_") -> "Config":
        """Load config from environment variables."""
        data = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Type conversion
                if config_key in ("verbose", "memory_persistence"):
                    data[config_key] = value.lower() in ("true", "1", "yes")
                elif config_key in ("max_tokens", "max_iterations", "memory_capacity"):
                    data[config_key] = int(value)
                elif config_key in ("temperature", "top_p", "tool_timeout"):
                    data[config_key] = float(value)
                else:
                    data[config_key] = value
        
        return cls.from_dict(data)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "api_key": self.api_key[:8] + "..." if self.api_key else None,
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_iterations": self.max_iterations,
            "tool_timeout": self.tool_timeout,
            "memory_capacity": self.memory_capacity,
            "memory_persistence": self.memory_persistence,
            "verbose": self.verbose,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "cache_dir": self.cache_dir,
            "extra": self.extra,
        }
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't save API key to file
        data = self.to_dict()
        data["api_key"] = None
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a config value."""
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            self.extra[key] = value
    
    def copy(self, **overrides) -> "Config":
        """Create a copy with overrides."""
        data = self.to_dict()
        data.update(overrides)
        data["api_key"] = overrides.get("api_key", self.api_key)
        return self.from_dict(data)
    
    def __repr__(self) -> str:
        return (
            f"Config(llm_provider={self.llm_provider}, "
            f"model={self.llm_model}, verbose={self.verbose})"
        )


# Global default config
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global default config."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Set the global default config."""
    global _default_config
    _default_config = config


def load_config(
    file_path: Optional[str] = None,
    from_env: bool = True,
    **overrides,
) -> Config:
    """
    Load configuration from multiple sources.
    
    Priority (highest to lowest):
    1. Overrides (keyword arguments)
    2. Environment variables
    3. Config file
    4. Defaults
    
    Args:
        file_path: Optional path to config file
        from_env: Whether to load from environment variables
        **overrides: Additional config values
        
    Returns:
        Merged configuration
    """
    # Start with defaults
    config = Config()
    
    # Load from file
    if file_path and os.path.exists(file_path):
        file_config = Config.from_file(file_path)
        for key, value in file_config.to_dict().items():
            if value is not None:
                setattr(config, key, value)
    
    # Load from environment
    if from_env:
        env_config = Config.from_env()
        for key, value in env_config.to_dict().items():
            if value is not None:
                setattr(config, key, value)
    
    # Apply overrides
    for key, value in overrides.items():
        if key in config.__dataclass_fields__:
            setattr(config, key, value)
        else:
            config.extra[key] = value
    
    return config
