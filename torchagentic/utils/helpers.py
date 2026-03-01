"""
Helper functions for TorchAgentic.

Provides common utilities for async operations, text processing, and more.
"""

import asyncio
import functools
import textwrap
from typing import Any, AsyncGenerator, Callable, Generator, Optional, TypeVar

T = TypeVar("T")


def run_async(coro: asyncio.coroutines.Coroutine) -> Any:
    """
    Run an async coroutine from synchronous code.
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def retry_async(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying async functions.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def timeout_async(timeout: float) -> Callable:
    """
    Decorator for adding timeout to async functions.
    
    Args:
        timeout: Timeout in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator


def batch_iterate(
    items: list[T],
    batch_size: int,
) -> Generator[list[T], None, None]:
    """
    Iterate over items in batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


async def async_batch_iterate(
    items: list[T],
    batch_size: int,
) -> AsyncGenerator[list[T], None]:
    """
    Async version of batch_iterate.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for batch in batch_iterate(items, batch_size):
        yield batch


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 0) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at word boundary
        if end < len(text) and overlap == 0:
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx + 1
        
        chunk = text[start:end]
        
        # Add overlap
        if overlap > 0 and chunks:
            prev_end = start
            start = max(0, start - overlap)
            chunk = text[start:end]
        
        chunks.append(chunk)
        start = end
    
    return chunks


def format_tokens(tokens: int) -> str:
    """
    Format token count as human-readable string.
    
    Args:
        tokens: Number of tokens
        
    Returns:
        Formatted string
    """
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    else:
        return str(tokens)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def indent_text(text: str, spaces: int = 4) -> str:
    """
    Add indentation to each line of text.
    
    Args:
        text: Text to indent
        spaces: Number of spaces for indentation
        
    Returns:
        Indented text
    """
    return textwrap.indent(text, " " * spaces)


def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting from text.
    
    Args:
        text: Text with markdown
        
    Returns:
        Plain text
    """
    import re
    
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    
    # Remove headers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    
    # Remove bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    
    # Remove links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    
    # Remove blockquotes
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    
    # Remove lists
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = text.strip()
    
    return text


def extract_code_blocks(text: str) -> list[dict[str, str]]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text: Markdown text
        
    Returns:
        List of dicts with 'language' and 'code' keys
    """
    import re
    
    pattern = r"```(\w+)?\s*\n([\s\S]*?)```"
    matches = re.findall(pattern, text)
    
    return [
        {"language": lang or "text", "code": code.strip()}
        for lang, code in matches
    ]


def merge_dicts(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def safe_get(d: dict, *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        d: Dictionary
        *keys: Keys to traverse
        default: Default value if not found
        
    Returns:
        Value or default
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d
