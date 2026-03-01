"""
Workflow Example

This example demonstrates how to create and execute workflows
with task dependencies.
"""

import asyncio
import time
from torchagentic.workflows import Workflow, Orchestrator, TaskStatus
from torchagentic import Agent
from torchagentic.llms.local import MockLLM


# Sample task functions
async def load_data(context: dict, **kwargs):
    """Simulate loading data."""
    await asyncio.sleep(0.1)
    print("  [Task] Loading data...")
    return {"rows": 1000, "columns": 10}


async def validate_data(context: dict, dep_load_data: dict = None, **kwargs):
    """Simulate data validation."""
    await asyncio.sleep(0.1)
    print(f"  [Task] Validating {dep_load_data.get('rows', 0)} rows...")
    return {"valid": True, "errors": []}


async def transform_data(context: dict, dep_validate_data: dict = None, **kwargs):
    """Simulate data transformation."""
    await asyncio.sleep(0.1)
    print("  [Task] Transforming data...")
    return {"transformed": True, "new_columns": 15}


async def save_results(context: dict, dep_transform_data: dict = None, **kwargs):
    """Simulate saving results."""
    await asyncio.sleep(0.1)
    print("  [Task] Saving results to database...")
    return {"saved": True, "location": "db://results/table1"}


async def send_notification(context: dict, dep_save_results: dict = None, **kwargs):
    """Simulate sending notification."""
    await asyncio.sleep(0.1)
    print("  [Task] Sending completion notification...")
    return {"sent": True, "recipients": ["user@example.com"]}


async def generate_report(context: dict, dep_transform_data: dict = None, **kwargs):
    """Simulate report generation."""
    await asyncio.sleep(0.1)
    print("  [Task] Generating PDF report...")
    return {"report_path": "/reports/summary.pdf"}


async def main():
    print("=" * 50)
    print("TorchAgentic - Workflow Example")
    print("=" * 50)
    
    print("\n[1] Creating a Data Processing Workflow")
    print("-" * 30)
    
    # Create workflow
    workflow = Workflow(
        name="Data Processing Pipeline",
        description="A complete data processing workflow",
    )
    
    # Add tasks with dependencies
    load_id = workflow.add_task(
        name="Load Data",
        func=load_data,
    )
    
    validate_id = workflow.add_task(
        name="Validate Data",
        func=validate_data,
        dependencies=["Load Data"],
    )
    
    transform_id = workflow.add_task(
        name="Transform Data",
        func=transform_data,
        dependencies=["Validate Data"],
    )
    
    save_id = workflow.add_task(
        name="Save Results",
        func=save_results,
        dependencies=["Transform Data"],
    )
    
    # Parallel tasks after transform
    notify_id = workflow.add_task(
        name="Send Notification",
        func=send_notification,
        dependencies=["Save Results"],
    )
    
    report_id = workflow.add_task(
        name="Generate Report",
        func=generate_report,
        dependencies=["Transform Data"],  # Can run parallel with Save Results
    )
    
    print(f"Created workflow with {len(workflow.tasks)} tasks")
    
    # Visualize workflow
    print("\nWorkflow visualization:")
    print(workflow.visualize())
    
    print("\n[2] Executing Workflow (Sequential)")
    print("-" * 30)
    
    # Execute sequentially
    start_time = time.time()
    results = await workflow.execute(parallel=False)
    sequential_time = time.time() - start_time
    
    print(f"\nSequential execution completed in {sequential_time:.2f}s")
    print("Task results:")
    for task in workflow.tasks.values():
        result = results.get(task.id)
        status_icon = "✅" if result and result.is_success else "❌"
        print(f"  {status_icon} {task.name}: {task.status.value}")
    
    # Reset workflow for parallel execution
    workflow.reset()
    
    print("\n[3] Executing Workflow (Parallel)")
    print("-" * 30)
    
    # Execute in parallel
    start_time = time.time()
    results = await workflow.execute(parallel=True)
    parallel_time = time.time() - start_time
    
    print(f"\nParallel execution completed in {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    
    print("\nTask results:")
    for task in workflow.tasks.values():
        result = results.get(task.id)
        if result and result.result:
            print(f"  ✅ {task.name}: {result.result}")
    
    print("\n[4] Using Orchestrator")
    print("-" * 30)
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Create and register workflow through orchestrator
    workflow2 = orchestrator.create_workflow(
        name="Simple Workflow",
        description="A simple workflow managed by orchestrator",
    )
    
    async def task_a(context, **kwargs):
        print("  Task A executed")
        return {"result": "A"}
    
    async def task_b(context, **kwargs):
        print("  Task B executed")
        return {"result": "B"}
    
    workflow2.add_task("Task A", func=task_a)
    workflow2.add_task("Task B", func=task_b, dependencies=["Task A"])
    
    # Execute through orchestrator
    print("Executing workflow through orchestrator...")
    results = await orchestrator.execute_workflow(workflow2.id)
    
    # Get orchestrator stats
    print("\nOrchestrator statistics:")
    stats = orchestrator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n[5] Workflow with Agent Tasks")
    print("-" * 30)
    
    # Create agents for different tasks
    analysis_llm = MockLLM(response="Analysis: The data looks good.")
    summary_llm = MockLLM(response="Summary: All tasks completed successfully.")
    
    analysis_agent = Agent(llm=analysis_llm, name="AnalysisAgent")
    summary_agent = Agent(llm=summary_llm, name="SummaryAgent")
    
    # Create workflow with agent tasks
    agent_workflow = Workflow(name="Agent Collaboration")
    
    async def analyze_task(context, **kwargs):
        response = await analysis_agent.act("Analyze the data")
        return response.content
    
    async def summarize_task(context, dep_analyze=None, **kwargs):
        response = await summary_agent.act(f"Summarize: {dep_analyze}")
        return response.content
    
    agent_workflow.add_task("Analyze", func=analyze_task)
    agent_workflow.add_task("Summarize", func=summarize_task, dependencies=["Analyze"])
    
    print("Created agent collaboration workflow")
    results = await agent_workflow.execute()
    
    for task in agent_workflow.tasks.values():
        result = results.get(task.id)
        if result:
            print(f"  {task.name}: {result.result}")
    
    print("\n[6] Error Handling")
    print("-" * 30)
    
    # Create workflow with a failing task
    error_workflow = Workflow(name="Error Handling Demo")
    
    async def good_task(context, **kwargs):
        return "Success"
    
    async def failing_task(context, **kwargs):
        raise ValueError("This task always fails!")
    
    async def dependent_task(context, dep_failing=None, **kwargs):
        return "Should not reach here"
    
    error_workflow.add_task("Good Task", func=good_task)
    error_workflow.add_task("Failing Task", func=failing_task, dependencies=["Good Task"])
    error_workflow.add_task("Dependent Task", func=dependent_task, dependencies=["Failing Task"])
    
    print("Executing workflow with failing task...")
    try:
        results = await error_workflow.execute()
    except Exception as e:
        print(f"Caught expected error: {e}")
    
    print("\nTask statuses after error:")
    for task in error_workflow.tasks.values():
        print(f"  {task.name}: {task.status.value}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
