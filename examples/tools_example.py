"""
Tools Example

This example demonstrates how to create and use tools with agents.
"""

import asyncio
from torchagentic import Agent, tool, Tool, ToolParameter
from torchagentic.llms.local import MockLLM


# Method 1: Using the @tool decorator
@tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool(tags=["math", "advanced"])
def power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent."""
    return base ** exponent


# Method 2: Creating a custom Tool class
class WeatherTool(Tool):
    """Tool for getting weather information (simulated)."""
    
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a city",
            tags=["weather", "api"],
        )
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="city",
                type="string",
                description="Name of the city",
                required=True,
            ),
            ToolParameter(
                name="units",
                type="string",
                description="Temperature units (celsius or fahrenheit)",
                required=False,
                default="celsius",
            ),
        ]
    
    async def execute(self, city: str, units: str = "celsius") -> str:
        # Simulated weather data
        weather_data = {
            "new york": {"temp_c": 22, "condition": "Partly cloudy"},
            "london": {"temp_c": 15, "condition": "Rainy"},
            "tokyo": {"temp_c": 28, "condition": "Sunny"},
        }
        
        city_lower = city.lower()
        if city_lower in weather_data:
            data = weather_data[city_lower]
            temp = data["temp_c"]
            if units == "fahrenheit":
                temp = (temp * 9/5) + 32
            return f"Weather in {city}: {data['condition']}, {temp}°{units[0].upper()}"
        else:
            return f"Weather data not available for {city}"


class CalculatorTool(Tool):
    """Advanced calculator tool."""
    
    def __init__(self):
        super().__init__(
            name="advanced_calculator",
            description="Perform complex mathematical calculations",
            tags=["math", "calculator"],
        )
    
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate",
                required=True,
            ),
        ]
    
    async def execute(self, expression: str) -> str:
        import math
        
        # Safe evaluation
        safe_dict = {
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_")
        }
        
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"


async def main():
    print("=" * 50)
    print("TorchAgentic - Tools Example")
    print("=" * 50)
    
    # Create LLM
    llm = MockLLM(response="I'll use my tools to help you with that.")
    
    # Create agent with tools
    agent = Agent(
        llm=llm,
        name="ToolAgent",
        system_prompt="You are a helpful assistant with access to various tools.",
        tools=[
            add,
            multiply,
            power,
            WeatherTool(),
            CalculatorTool(),
        ],
        verbose=True,
    )
    
    # List available tools
    print("\n[1] Available tools:")
    for tool_name, tool_obj in agent.tool_manager.tools.items():
        print(f"  - {tool_name}: {tool_obj.description[:50]}...")
    
    # Demonstrate tool execution
    print("\n[2] Direct tool execution:")
    
    # Execute add tool
    result = await agent.tool_manager.execute("add", {"a": 5, "b": 3})
    print(f"  add(5, 3) = {result['result']}")
    
    # Execute multiply tool
    result = await agent.tool_manager.execute("multiply", {"a": 4, "b": 7})
    print(f"  multiply(4, 7) = {result['result']}")
    
    # Execute power tool
    result = await agent.tool_manager.execute("power", {"base": 2, "exponent": 10})
    print(f"  power(2, 10) = {result['result']}")
    
    # Execute weather tool
    result = await agent.tool_manager.execute("get_weather", {"city": "Tokyo"})
    print(f"  get_weather('Tokyo') = {result['result']}")
    
    # Execute calculator tool
    result = await agent.tool_manager.execute("advanced_calculator", {
        "expression": "sqrt(144) + log(100)"
    })
    print(f"  advanced_calculator('sqrt(144) + log(100)') = {result['result']}")
    
    # Show tool statistics
    print("\n[3] Tool execution statistics:")
    stats = agent.tool_manager.get_statistics()
    for tool_name, tool_stats in stats.items():
        print(f"  {tool_name}:")
        print(f"    Executions: {tool_stats['execution_count']}")
        print(f"    Avg time: {tool_stats['avg_execution_time']:.4f}s")
    
    # Show tool schema (for LLM)
    print("\n[4] Tool schema for LLM:")
    schema = agent.tool_manager.get_schema()
    for tool_schema in schema:
        func = tool_schema.get("function", {})
        print(f"  - {func.get('name', 'unknown')}: {func.get('description', '')[:40]}...")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
