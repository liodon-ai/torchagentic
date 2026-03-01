"""
Command-line interface for TorchAgentic.

Provides basic CLI utilities for interacting with agents and workflows.
"""

import argparse
import asyncio
import json
import sys
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="torchagentic",
        description="TorchAgentic - AI Agent Framework",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with an agent")
    chat_parser.add_argument(
        "-m", "--model",
        default="microsoft/phi-2",
        help="Model to use for chat",
    )
    chat_parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    chat_parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the agent",
    )
    chat_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    # Run workflow command
    workflow_parser = subparsers.add_parser(
        "run",
        help="Run a workflow from a JSON file",
    )
    workflow_parser.add_argument(
        "workflow_file",
        type=str,
        help="Path to workflow JSON file",
    )
    workflow_parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="JSON string with initial context",
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show framework information")
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    return parser


async def chat_command(args: argparse.Namespace) -> None:
    """Handle the chat command."""
    from torchagentic import Agent
    from torchagentic.llms.local import LocalLLM
    
    print(f"Loading model: {args.model}")
    llm = LocalLLM(model_id=args.model)
    
    agent = Agent(
        llm=llm,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        verbose=args.verbose,
    )
    
    print("TorchAgentic Chat (type 'quit' or 'exit' to stop)")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("Agent: ", end="", flush=True)
            response = await agent.act(user_input)
            print(response.content or "[No response]")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


async def run_workflow_command(args: argparse.Namespace) -> None:
    """Handle the run workflow command."""
    from torchagentic.workflows.orchestrator import Orchestrator
    
    # Load workflow
    with open(args.workflow_file, "r") as f:
        workflow_data = json.load(f)
    
    # Parse context
    context = {}
    if args.context:
        context = json.loads(args.context)
    
    # Create orchestrator and workflow
    orchestrator = Orchestrator()
    
    # TODO: Implement workflow loading from JSON
    print("Workflow execution from JSON is not yet implemented.")
    print("Please use the Python API to create and run workflows.")


def info_command(args: argparse.Namespace) -> None:
    """Handle the info command."""
    from torchagentic import __version__
    import torch
    
    print("TorchAgentic Information")
    print("=" * 40)
    print(f"TorchAgentic version: {__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def version_command(args: argparse.Namespace) -> None:
    """Handle the version command."""
    from torchagentic import __version__
    print(f"torchagentic {__version__}")


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "chat":
        asyncio.run(chat_command(args))
    elif args.command == "run":
        asyncio.run(run_workflow_command(args))
    elif args.command == "info":
        info_command(args)
    elif args.command == "version":
        version_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
