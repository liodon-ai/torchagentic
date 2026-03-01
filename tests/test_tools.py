"""
Tests for tools module.
"""

import pytest
from torchagentic.tools.base import Tool, ToolRegistry, FunctionTool, tool, ToolParameter
from torchagentic.tools.manager import ToolManager
from torchagentic.tools.builtin import CalculatorTool, DateTimeTool


class TestToolDecorator:
    """Tests for @tool decorator."""
    
    def test_tool_decorator_basic(self):
        @tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        assert isinstance(add, FunctionTool)
        assert add.name == "add"
    
    def test_tool_decorator_with_name(self):
        @tool(name="custom_add")
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        assert add.name == "custom_add"
    
    def test_tool_decorator_with_tags(self):
        @tool(tags=["math", "utility"])
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        assert "math" in add.tags
        assert "utility" in add.tags
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        @tool()
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b
        
        result = await multiply.execute(a=3, b=4)
        assert result == 12
    
    @pytest.mark.asyncio
    async def test_async_tool(self):
        @tool()
        async def fetch_data(url: str) -> str:
            """Fetch data from URL."""
            return f"Data from {url}"
        
        result = await fetch_data.execute(url="http://example.com")
        assert result == "Data from http://example.com"


class TestToolClass:
    """Tests for Tool base class."""
    
    def test_custom_tool(self):
        class CustomTool(Tool):
            def __init__(self):
                super().__init__(
                    name="custom",
                    description="A custom tool",
                )
            
            def get_parameters(self):
                return [
                    ToolParameter("input", "string", "Input value"),
                ]
            
            async def execute(self, input: str):
                return f"Processed: {input}"
        
        tool = CustomTool()
        assert tool.name == "custom"
        assert len(tool.get_parameters()) == 1
    
    def test_tool_to_dict(self):
        class TestTool(Tool):
            def __init__(self):
                super().__init__(name="test", description="Test tool")
            
            def get_parameters(self):
                return [
                    ToolParameter("x", "integer", "Number", required=True),
                ]
            
            async def execute(self, x: int):
                return x
        
        tool = TestTool()
        data = tool.to_dict()
        assert data["name"] == "test"
        assert "x" in data["parameters"]["properties"]
    
    def test_tool_openai_schema(self):
        class TestTool(Tool):
            def __init__(self):
                super().__init__(name="test", description="Test")
            
            def get_parameters(self):
                return [
                    ToolParameter("name", "string", "Name", required=True),
                ]
            
            async def execute(self, name: str):
                return name
        
        tool = TestTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema


class TestToolRegistry:
    """Tests for ToolRegistry class."""
    
    def test_register_tool(self):
        registry = ToolRegistry()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        registry.register(add)
        assert registry.has("add")
        assert len(registry) == 1
    
    def test_unregister_tool(self):
        registry = ToolRegistry()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        registry.register(add)
        registry.unregister("add")
        assert not registry.has("add")
    
    def test_get_tool(self):
        registry = ToolRegistry()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        registry.register(add)
        tool = registry.get("add")
        assert tool is not None
        assert tool.name == "add"
    
    def test_list_tools(self):
        registry = ToolRegistry()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        @tool()
        def sub(a: int, b: int) -> int:
            return a - b
        
        registry.register(add)
        registry.register(sub)
        
        tools = registry.list_tools()
        assert "add" in tools
        assert "sub" in tools
    
    def test_get_by_tag(self):
        registry = ToolRegistry()
        
        @tool(tags=["math"])
        def add(a: int, b: int) -> int:
            return a + b
        
        @tool(tags=["string"])
        def upper(s: str) -> str:
            return s.upper()
        
        registry.register(add)
        registry.register(upper)
        
        math_tools = registry.get_by_tag("math")
        assert len(math_tools) == 1
        assert math_tools[0].name == "add"
    
    def test_get_schema(self):
        registry = ToolRegistry()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        registry.register(add)
        schema = registry.get_schema()
        assert len(schema) == 1
        assert schema[0]["type"] == "function"


class TestToolManager:
    """Tests for ToolManager class."""
    
    def test_register_and_execute(self):
        manager = ToolManager()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        manager.register(add)
        
        import asyncio
        result = asyncio.run(manager.execute("add", {"a": 1, "b": 2}))
        assert result["success"]
        assert result["result"] == 3
    
    def test_execute_unknown_tool(self):
        manager = ToolManager()
        
        import asyncio
        result = asyncio.run(manager.execute("unknown", {}))
        assert not result["success"]
        assert "not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_batch_execute(self):
        manager = ToolManager()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        @tool()
        def mul(a: int, b: int) -> int:
            return a * b
        
        manager.register(add)
        manager.register(mul)
        
        results = await manager.execute_batch([
            {"name": "add", "arguments": {"a": 1, "b": 2}},
            {"name": "mul", "arguments": {"a": 3, "b": 4}},
        ])
        
        assert len(results) == 2
        assert results[0]["result"] == 3
        assert results[1]["result"] == 12
    
    def test_tool_statistics(self):
        manager = ToolManager()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        manager.register(add)
        
        import asyncio
        asyncio.run(manager.execute("add", {"a": 1, "b": 2}))
        asyncio.run(manager.execute("add", {"a": 3, "b": 4}))
        
        stats = manager.get_statistics("add")
        assert stats["execution_count"] == 2
    
    def test_reset_statistics(self):
        manager = ToolManager()
        
        @tool()
        def add(a: int, b: int) -> int:
            return a + b
        
        manager.register(add)
        
        import asyncio
        asyncio.run(manager.execute("add", {"a": 1, "b": 2}))
        
        manager.reset_statistics("add")
        stats = manager.get_statistics("add")
        assert stats["execution_count"] == 0


class TestBuiltinTools:
    """Tests for built-in tools."""
    
    @pytest.mark.asyncio
    async def test_calculator_tool(self):
        tool = CalculatorTool()
        result = await tool.execute(expression="2 + 3 * 4")
        assert "14" in result
    
    @pytest.mark.asyncio
    async def test_datetime_tool(self):
        tool = DateTimeTool()
        result = await tool.execute(component="year")
        assert len(result) == 4  # Year is 4 digits
    
    @pytest.mark.asyncio
    async def test_datetime_tool_format(self):
        tool = DateTimeTool()
        result = await tool.execute(format="%Y-%m-%d")
        assert "-" in result
