"""
Multi-Agent Example

This example demonstrates how to create and coordinate multiple agents
working together on a task.
"""

import asyncio
from torchagentic import Agent
from torchagentic.workflows import MultiAgentOrchestrator
from torchagentic.llms.local import MockLLM


async def main():
    print("=" * 50)
    print("TorchAgentic - Multi-Agent Example")
    print("=" * 50)
    
    print("\n[1] Creating Multiple Agents")
    print("-" * 30)
    
    # Create specialized agents with different personas
    researcher_llm = MockLLM(
        response="Research: I found relevant information about the topic. "
                 "Key findings include: 1) Market size is growing, "
                 "2) New technologies emerging, 3) Competition increasing."
    )
    analyst_llm = MockLLM(
        response="Analysis: Based on the research, I analyze that the trends "
                 "are positive. Growth rate ~15% YoY. Risks include market saturation."
    )
    writer_llm = MockLLM(
        response="Report: Here's the final report:\n\n"
                 "# Market Analysis Report\n\n"
                 "## Executive Summary\nThe market shows strong growth.\n\n"
                 "## Findings\nPositive trends identified.\n\n"
                 "## Recommendations\nInvest now."
    )
    
    researcher = Agent(
        llm=researcher_llm,
        name="Researcher",
        system_prompt="You are a research specialist. Find and summarize information.",
    )
    
    analyst = Agent(
        llm=analyst_llm,
        name="Analyst",
        system_prompt="You are a data analyst. Analyze information and identify trends.",
    )
    
    writer = Agent(
        llm=writer_llm,
        name="Writer",
        system_prompt="You are a technical writer. Create clear, well-structured reports.",
    )
    
    print(f"Created {3} specialized agents:")
    for agent in [researcher, analyst, writer]:
        print(f"  - {agent.name}: {agent.system_prompt[:50]}...")
    
    print("\n[2] Using Multi-Agent Orchestrator")
    print("-" * 30)
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Register agents
    orchestrator.register_agents([researcher, analyst, writer])
    
    print(f"Registered {len(orchestrator.agents)} agents with orchestrator")
    
    # Show agent stats
    stats = orchestrator.get_agent_stats()
    print(f"\nAgent statistics:")
    print(f"  Total agents: {stats['total_agents']}")
    
    print("\n[3] Collaborative Task Execution")
    print("-" * 30)
    
    # Execute collaborative task
    task_description = """
    Create a market analysis report for AI technologies.
    
    Process:
    1. Research current market trends
    2. Analyze the data and identify opportunities
    3. Write a comprehensive report
    """
    
    print(f"Task: {task_description.strip()[:100]}...")
    print("\nStarting collaborative execution...")
    
    results = await orchestrator.execute_collaborative_task(
        task_description=task_description,
        max_rounds=3,
    )
    
    print(f"\nCollaboration completed!")
    print(f"  Agents involved: {', '.join(results['agents_involved'])}")
    print(f"  Rounds completed: {results['rounds_completed']}")
    
    print("\n[4] Agent-to-Agent Messaging")
    print("-" * 30)
    
    # Demonstrate direct messaging
    print("Sending messages between agents...")
    
    await orchestrator.send_message(
        from_agent_id=researcher.id,
        to_agent_id=analyst.id,
        message="I found some interesting data about AI market growth.",
    )
    
    await orchestrator.send_message(
        from_agent_id=analyst.id,
        to_agent_id=writer.id,
        message="The analysis shows 15% YoY growth. Please include this in the report.",
    )
    
    # Check messages
    print("\nChecking messages for each agent:")
    for agent in [researcher, analyst, writer]:
        messages = orchestrator.get_messages(agent.id, clear=False)
        if messages:
            print(f"  {agent.name} has {len(messages)} message(s)")
            for msg in messages:
                from_agent = orchestrator.get_agent(msg["from"])
                print(f"    From {from_agent.name}: {msg['content'][:50]}...")
        else:
            print(f"  {agent.name}: No messages")
    
    print("\n[5] Creating Agent Workflow")
    print("-" * 30)
    
    # Create a workflow where agents work in sequence
    agent_workflow = await orchestrator.create_agent_workflow(
        name="Report Generation Pipeline",
        agent_tasks=[
            {"agent": "Researcher", "name": "Research", "input": "Research AI market trends"},
            {"agent": "Analyst", "name": "Analyze", "input": "Analyze the research findings"},
            {"agent": "Writer", "name": "Write Report", "input": "Write a report based on the analysis"},
        ],
    )
    
    print(f"Created workflow: {agent_workflow.name}")
    print(f"Tasks: {len(agent_workflow.tasks)}")
    
    # Visualize
    print("\nWorkflow visualization:")
    print(agent_workflow.visualize())
    
    # Execute
    print("\nExecuting agent workflow...")
    results = await agent_workflow.execute()
    
    print("\nWorkflow results:")
    for task in agent_workflow.tasks.values():
        result = results.get(task.id)
        if result and result.result:
            content = str(result.result)[:100] if hasattr(result.result, 'content') else str(result.result)[:100]
            print(f"  ✅ {task.name}: {content}...")
    
    print("\n[6] Orchestrator Statistics")
    print("-" * 30)
    
    # Get final statistics
    print("Final orchestrator state:")
    orchestrator_dict = orchestrator.to_dict()
    
    print(f"  Workflows: {len(orchestrator_dict['workflows'])}")
    print(f"  Agents: {len(orchestrator_dict['agents'])}")
    
    stats = orchestrator.get_stats()
    print(f"\nExecution statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Successful: {stats['successful_executions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
