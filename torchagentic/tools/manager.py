"""
Tool manager for executing and coordinating tools.
"""

from typing import Any, Optional
import asyncio
import time

from torchagentic.tools.base import Tool, ToolRegistry, FunctionTool


class ToolManager:
    """
    Manager for tool execution.
    
    Handles registration, execution, and error handling for tools.
    
    Attributes:
        registry: The underlying tool registry
        timeout: Default timeout for tool execution in seconds
        max_concurrent: Maximum concurrent tool executions
    """
    
    def __init__(
        self,
        tools: Optional[list[Tool | FunctionTool | callable]] = None,
        timeout: float = 30.0,
        max_concurrent: int = 5,
    ):
        self.registry = ToolRegistry()
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        
        # Track execution statistics
        self._execution_count: dict[str, int] = {}
        self._total_execution_time: dict[str, float] = {}
        
        if tools:
            for tool in tools:
                self.register(tool)
    
    def register(self, tool: Tool | FunctionTool | callable) -> None:
        """
        Register a tool with the manager.
        
        Args:
            tool: Tool instance, FunctionTool, or callable function
        """
        if callable(tool) and not isinstance(tool, Tool):
            tool = FunctionTool(func=tool)
        
        if isinstance(tool, Tool):
            self.registry.register(tool)
            self._execution_count[tool.name] = 0
            self._total_execution_time[tool.name] = 0.0
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._execution_count:
            del self._execution_count[name]
        if name in self._total_execution_time:
            del self._total_execution_time[name]
        return self.registry.unregister(name)
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.registry.get(name)
    
    @property
    def tools(self) -> dict[str, Tool]:
        """Get all registered tools."""
        return self.registry._tools
    
    def get_schema(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for all tools."""
        return self.registry.get_schema()
    
    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            timeout: Optional timeout override
        
        Returns:
            Dictionary with 'result' or 'error' key
        """
        tool = self.registry.get(name)
        
        if tool is None:
            return {
                "success": False,
                "error": f"Tool not found: {name}",
                "available_tools": self.registry.list_tools(),
            }
        
        start_time = time.time()
        exec_timeout = timeout or self.timeout
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=exec_timeout,
            )
            
            # Update statistics
            execution_time = time.time() - start_time
            self._execution_count[name] = self._execution_count.get(name, 0) + 1
            self._total_execution_time[name] = (
                self._total_execution_time.get(name, 0) + execution_time
            )
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Tool execution timed out after {exec_timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }
    
    async def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Execute multiple tools in parallel.
        
        Args:
            tool_calls: List of dicts with 'name' and 'arguments' keys
        
        Returns:
            List of execution results
        """
        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def execute_with_semaphore(call: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                result = await self.execute(
                    call.get("name", ""),
                    call.get("arguments", {}),
                )
                result["tool_call_id"] = call.get("id")
                return result
        
        tasks = [execute_with_semaphore(call) for call in tool_calls]
        return await asyncio.gather(*tasks)
    
    def get_statistics(self, name: Optional[str] = None) -> dict[str, Any]:
        """
        Get execution statistics.
        
        Args:
            name: Optional tool name (returns all stats if None)
        
        Returns:
            Dictionary with execution statistics
        """
        if name:
            return {
                "name": name,
                "execution_count": self._execution_count.get(name, 0),
                "total_execution_time": self._total_execution_time.get(name, 0),
                "avg_execution_time": (
                    self._total_execution_time.get(name, 0) /
                    max(1, self._execution_count.get(name, 1))
                ),
            }
        
        return {
            name: self.get_statistics(name)
            for name in self._execution_count.keys()
        }
    
    def reset_statistics(self, name: Optional[str] = None) -> None:
        """
        Reset execution statistics.
        
        Args:
            name: Optional tool name (resets all if None)
        """
        if name:
            self._execution_count[name] = 0
            self._total_execution_time[name] = 0
        else:
            self._execution_count = {k: 0 for k in self._execution_count}
            self._total_execution_time = {k: 0 for k in self._total_execution_time}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert manager to dictionary format."""
        return {
            "tools": self.registry.to_dict(),
            "statistics": self.get_statistics(),
            "config": {
                "timeout": self.timeout,
                "max_concurrent": self.max_concurrent,
            },
        }
    
    def __repr__(self) -> str:
        return f"ToolManager(tools={self.registry.list_tools()})"
