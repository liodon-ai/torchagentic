# TorchAgentic

A PyTorch-based library for building AI agents and agentic workflows.

## Overview

TorchAgentic provides a comprehensive framework for creating, managing, and orchestrating AI agents with support for:

- 🤖 **Agent System** - Create intelligent agents powered by LLMs
- 🔧 **Tool Calling** - Equip agents with custom tools and functions
- 🧠 **Memory Management** - Short-term and long-term memory systems
- ⚙️ **Workflow Orchestration** - Multi-step task automation
- 🔄 **Multi-Agent Systems** - Collaborative agent workflows
- 🏃 **Local LLM Support** - Run models locally with PyTorch

## Installation

### Basic Installation

```bash
pip install torchagentic
```

### With LLM Support

```bash
pip install torchagentic[llm]
```

### Full Installation

```bash
pip install torchagentic[full]
```

### Development Installation

```bash
pip install torchagentic[dev]
```

## Quick Start

### Creating a Simple Agent

```python
import asyncio
from torchagentic import Agent
from torchagentic.llms.local import LocalLLM

async def main():
    # Create a local LLM
    llm = LocalLLM(model_id="microsoft/phi-2")
    
    # Create an agent
    agent = Agent(
        llm=llm,
        system_prompt="You are a helpful assistant.",
        verbose=True,
    )
    
    # Chat with the agent
    response = await agent.act("Hello! What can you do?")
    print(response.content)

asyncio.run(main())
```

### Adding Tools to an Agent

```python
from torchagentic import Agent, tool
from torchagentic.llms.local import LocalLLM

# Define a custom tool
@tool()
def calculate_bmi(weight: float, height: float) -> float:
    """Calculate BMI from weight (kg) and height (m)."""
    return weight / (height ** 2)

# Create agent with tools
llm = LocalLLM(model_id="microsoft/phi-2")
agent = Agent(llm=llm, tools=[calculate_bmi])

# The agent can now use the calculate_bmi tool
response = await agent.act("Calculate BMI for weight 70kg and height 1.75m")
```

### Using Memory

```python
from torchagentic import Agent, ShortTermMemory, LongTermMemory
from torchagentic.llms.local import LocalLLM

llm = LocalLLM(model_id="microsoft/phi-2")

# Create memory systems
short_term = ShortTermMemory(capacity=100)
long_term = LongTermMemory(capacity=1000, storage_path="memory.json")

# Create agent with memory
agent = Agent(llm=llm, memory=short_term)
```

### Creating Workflows

```python
from torchagentic.workflows import Workflow, Orchestrator

# Create orchestrator
orchestrator = Orchestrator()

# Create workflow
workflow = orchestrator.create_workflow(name="Data Processing")

# Add tasks
@workflow.add_task(name="Load Data", func=load_data)
@workflow.add_task(name="Process Data", func=process_data, dependencies=["Load Data"])
@workflow.add_task(name="Save Results", func=save_results, dependencies=["Process Data"])

# Execute workflow
results = await workflow.execute()
```

## Core Components

### Agents

The `Agent` class is the core component for creating AI agents:

```python
from torchagentic import Agent

agent = Agent(
    llm=llm,                    # LLM to use
    name="my-agent",            # Agent name
    system_prompt="...",        # System prompt
    tools=[...],                # List of tools
    memory=memory,              # Memory system
    max_iterations=10,          # Max tool execution iterations
    temperature=0.7,            # LLM temperature
    verbose=True,               # Enable verbose output
)
```

### Tools

Tools extend agent capabilities:

```python
from torchagentic import tool, Tool

# Using decorator
@tool()
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return results

# Using class
class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="Does something useful",
        )
    
    async def execute(self, **kwargs):
        # Implementation
        return result
```

### Built-in Tools

- `CalculatorTool` - Mathematical calculations
- `DateTimeTool` - Date and time utilities
- `FileReadTool` / `FileWriteTool` - File operations
- `SearchTool` - Text search with regex
- `PythonREPLTool` - Python code execution
- `ShellCommandTool` - Shell command execution

### Memory

Two types of memory systems:

```python
from torchagentic import ShortTermMemory, LongTermMemory

# Short-term memory (fast, limited capacity)
short_term = ShortTermMemory(
    capacity=100,
    ttl=3600,  # 1 hour
)

# Long-term memory (persistent, semantic search)
long_term = LongTermMemory(
    capacity=1000,
    storage_path="memory.json",
)
```

### Workflows

Create multi-step automated workflows:

```python
from torchagentic.workflows import Workflow, Task

workflow = Workflow(name="My Workflow")

# Add tasks with dependencies
workflow.add_task(
    name="Step 1",
    func=step1_func,
)

workflow.add_task(
    name="Step 2",
    func=step2_func,
    dependencies=["Step 1"],
)

# Execute
results = await workflow.execute(parallel=True)
```

### Multi-Agent Systems

```python
from torchagentic.workflows import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator()

# Register agents
orchestrator.register_agents([agent1, agent2, agent3])

# Collaborative task
results = await orchestrator.execute_collaborative_task(
    task_description="Solve this problem together",
    max_rounds=5,
)
```

## LLM Integrations

### Local LLM

```python
from torchagentic.llms.local import LocalLLM

llm = LocalLLM(
    model_id="microsoft/phi-2",
    device="cuda",  # or "cpu", "mps"
)
```

### Mock LLM (for testing)

```python
from torchagentic.llms.local import MockLLM

llm = MockLLM(response="Mock response for testing")
```

## Configuration

```python
from torchagentic import Config

config = Config(
    llm_provider="local",
    llm_model="microsoft/phi-2",
    temperature=0.7,
    max_tokens=2048,
    verbose=True,
)

# Save config
config.save("config.json")

# Load config
config = Config.from_file("config.json")

# Load from environment
config = Config.from_env()
```

## CLI Usage

```bash
# Chat with an agent
torchagentic chat --model microsoft/phi-2

# Show information
torchagentic info

# Show version
torchagentic version
```

## Examples

See the `examples/` directory for complete examples:

- `basic_agent.py` - Simple agent creation
- `tools_example.py` - Using tools with agents
- `memory_example.py` - Memory systems
- `workflow_example.py` - Workflow orchestration
- `multi_agent.py` - Multi-agent collaboration

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Agent` | Main agent class |
| `BaseAgent` | Abstract base agent |
| `Message` | Conversation message |
| `Conversation` | Message history manager |
| `AgentResponse` | Agent execution response |

### Tool Classes

| Class | Description |
|-------|-------------|
| `Tool` | Base tool class |
| `FunctionTool` | Function-based tool |
| `ToolRegistry` | Tool registration |
| `ToolManager` | Tool execution manager |
| `@tool` | Tool decorator |

### Memory Classes

| Class | Description |
|-------|-------------|
| `BaseMemory` | Abstract memory base |
| `ShortTermMemory` | LRU memory |
| `LongTermMemory` | Persistent memory |
| `MemoryEntry` | Memory record |

### Workflow Classes

| Class | Description |
|-------|-------------|
| `Workflow` | Task workflow |
| `Task` | Workflow task |
| `Orchestrator` | Workflow manager |
| `MultiAgentOrchestrator` | Multi-agent manager |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Create a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- GitHub Issues: https://github.com/liodon-ai/torchagentic/issues
- Documentation: https://github.com/liodon-ai/torchagentic#readme
