"""
Tests for workflows module.
"""

import pytest
import asyncio
from torchagentic.workflows.workflow import Workflow, Task, TaskStatus, TaskResult
from torchagentic.workflows.orchestrator import Orchestrator, MultiAgentOrchestrator
from torchagentic import Agent
from torchagentic.llms.local import MockLLM


class TestTask:
    """Tests for Task class."""
    
    @pytest.mark.asyncio
    async def test_task_execution(self):
        async def my_func(context, **kwargs):
            return "Success"
        
        task = Task(name="Test", func=my_func)
        result = await task.execute({}, {})
        
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "Success"
    
    @pytest.mark.asyncio
    async def test_task_with_dependencies(self):
        async def task1(context, **kwargs):
            return "Result1"
        
        async def task2(context, dep_result=None, **kwargs):
            return f"Result2 with {dep_result}"
        
        t1 = Task(name="Task1", func=task1)
        t2 = Task(name="Task2", func=task2, dependencies=[t1.id])
        
        # Execute first task
        r1 = await t1.execute({}, {})
        
        # Execute second task with dependency result
        r2 = await t2.execute({}, {t1.id: r1})
        
        assert r2.status == TaskStatus.COMPLETED
        assert "Result1" in r2.result
    
    @pytest.mark.asyncio
    async def test_task_condition(self):
        async def my_func(context, **kwargs):
            return "Executed"
        
        # Condition that returns False
        task = Task(
            name="Test",
            func=my_func,
            condition=lambda ctx: ctx.get("run", False),
        )
        
        result = await task.execute({"run": False}, {})
        assert result.status == TaskStatus.SKIPPED
        
        # Condition that returns True
        task.reset()
        result = await task.execute({"run": True}, {})
        assert result.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_task_timeout(self):
        async def slow_func(context, **kwargs):
            await asyncio.sleep(10)
            return "Done"
        
        task = Task(name="Slow", func=slow_func, timeout=0.1)
        result = await task.execute({}, {})
        
        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.error
    
    @pytest.mark.asyncio
    async def test_task_retry(self):
        attempt_count = 0
        
        async def failing_func(context, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Fail")
            return "Success"
        
        task = Task(name="Retry", func=failing_func, retry_count=3)
        result = await task.execute({}, {})
        
        assert result.status == TaskStatus.COMPLETED
        assert attempt_count == 3
    
    def test_task_to_dict(self):
        async def my_func(context, **kwargs):
            return "Done"
        
        task = Task(name="Test", func=my_func)
        data = task.to_dict()
        
        assert data["name"] == "Test"
        assert data["status"] == "pending"


class TestWorkflow:
    """Tests for Workflow class."""
    
    @pytest.mark.asyncio
    async def test_create_workflow(self):
        workflow = Workflow(name="Test Workflow")
        assert workflow.name == "Test Workflow"
        assert len(workflow.tasks) == 0
    
    @pytest.mark.asyncio
    async def test_add_tasks(self):
        workflow = Workflow(name="Test")
        
        async def task1(context, **kwargs):
            return 1
        
        async def task2(context, **kwargs):
            return 2
        
        workflow.add_task("Task1", func=task1)
        workflow.add_task("Task2", func=task2, dependencies=["Task1"])
        
        assert len(workflow.tasks) == 2
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        workflow = Workflow(name="Test")
        
        results_store = []
        
        async def task1(context, **kwargs):
            results_store.append("task1")
            return "result1"
        
        async def task2(context, **kwargs):
            results_store.append("task2")
            return "result2"
        
        workflow.add_task("Task1", func=task1)
        workflow.add_task("Task2", func=task2)
        
        results = await workflow.execute()
        
        assert workflow.status == TaskStatus.COMPLETED
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_workflow_with_dependencies(self):
        workflow = Workflow(name="DepTest")
        
        execution_order = []
        
        async def task1(context, **kwargs):
            execution_order.append(1)
            return "r1"
        
        async def task2(context, **kwargs):
            execution_order.append(2)
            return "r2"
        
        workflow.add_task("Task1", func=task1)
        workflow.add_task("Task2", func=task2, dependencies=["Task1"])
        
        await workflow.execute()
        
        assert execution_order == [1, 2]
    
    @pytest.mark.asyncio
    async def test_workflow_parallel_execution(self):
        workflow = Workflow(name="ParallelTest")
        
        import time
        timestamps = {}
        
        async def task1(context, **kwargs):
            timestamps["task1"] = time.time()
            await asyncio.sleep(0.1)
            return "r1"
        
        async def task2(context, **kwargs):
            timestamps["task2"] = time.time()
            await asyncio.sleep(0.1)
            return "r2"
        
        workflow.add_task("Task1", func=task1)
        workflow.add_task("Task2", func=task2)
        
        await workflow.execute(parallel=True)
        
        # Tasks should start almost simultaneously
        time_diff = abs(timestamps["task1"] - timestamps["task2"])
        assert time_diff < 0.05  # Less than 50ms difference
    
    @pytest.mark.asyncio
    async def test_workflow_context(self):
        workflow = Workflow(name="ContextTest")
        
        async def use_context(context, **kwargs):
            return context.get("value", "default")
        
        workflow.add_task("Task", func=use_context)
        workflow.set_context("value", "custom")
        
        results = await workflow.execute()
        
        task = workflow.get_task_by_name("Task")
        result = results[task.id]
        assert result.result == "custom"
    
    @pytest.mark.asyncio
    async def test_workflow_get_result(self):
        workflow = Workflow(name="ResultTest")
        
        async def task1(context, **kwargs):
            return "important_result"
        
        workflow.add_task("Task1", func=task1)
        await workflow.execute()
        
        result = workflow.get_result("Task1")
        assert result == "important_result"
    
    @pytest.mark.asyncio
    async def test_workflow_reset(self):
        workflow = Workflow(name="ResetTest")
        
        async def task1(context, **kwargs):
            return "done"
        
        workflow.add_task("Task1", func=task1)
        await workflow.execute()
        
        assert workflow.status == TaskStatus.COMPLETED
        
        workflow.reset()
        assert workflow.status == TaskStatus.PENDING
    
    def test_workflow_visualize(self):
        workflow = Workflow(name="VizTest")
        
        async def task1(context, **kwargs):
            pass
        
        async def task2(context, **kwargs):
            pass
        
        workflow.add_task("Task1", func=task1)
        workflow.add_task("Task2", func=task2, dependencies=["Task1"])
        
        viz = workflow.visualize()
        assert "VizTest" in viz
        assert "Task1" in viz
        assert "Task2" in viz


class TestOrchestrator:
    """Tests for Orchestrator class."""
    
    @pytest.mark.asyncio
    async def test_create_orchestrator(self):
        orchestrator = Orchestrator()
        assert len(orchestrator.workflows) == 0
    
    @pytest.mark.asyncio
    async def test_register_workflow(self):
        orchestrator = Orchestrator()
        workflow = Workflow(name="Test")
        
        workflow_id = orchestrator.register_workflow(workflow)
        assert workflow_id == workflow.id
        assert len(orchestrator.workflows) == 1
    
    @pytest.mark.asyncio
    async def test_create_workflow(self):
        orchestrator = Orchestrator()
        workflow = orchestrator.create_workflow(name="New Workflow")
        
        assert workflow.name == "New Workflow"
        assert workflow.id in orchestrator.workflows
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        orchestrator = Orchestrator()
        
        workflow = orchestrator.create_workflow(name="ExecTest")
        
        async def task1(context, **kwargs):
            return context.get("input", "default")
        
        workflow.add_task("Task1", func=task1)
        
        results = await orchestrator.execute_workflow(
            workflow.id,
            context={"input": "custom"},
        )
        
        task = workflow.get_task_by_name("Task1")
        assert results[task.id].result == "custom"
    
    @pytest.mark.asyncio
    async def test_orchestrator_stats(self):
        orchestrator = Orchestrator()
        
        workflow = orchestrator.create_workflow(name="StatsTest")
        
        async def task1(context, **kwargs):
            return "done"
        
        workflow.add_task("Task1", func=task1)
        await orchestrator.execute_workflow(workflow.id)
        
        stats = orchestrator.get_stats()
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1


class TestMultiAgentOrchestrator:
    """Tests for MultiAgentOrchestrator class."""
    
    @pytest.mark.asyncio
    async def test_register_agents(self):
        orchestrator = MultiAgentOrchestrator()
        
        llm = MockLLM()
        agent1 = Agent(llm=llm, name="Agent1")
        agent2 = Agent(llm=llm, name="Agent2")
        
        orchestrator.register_agents([agent1, agent2])
        
        assert len(orchestrator.agents) == 2
    
    @pytest.mark.asyncio
    async def test_send_message(self):
        orchestrator = MultiAgentOrchestrator()
        
        llm = MockLLM()
        agent1 = Agent(llm=llm, name="Agent1")
        agent2 = Agent(llm=llm, name="Agent2")
        
        orchestrator.register_agents([agent1, agent2])
        
        await orchestrator.send_message(
            from_agent_id=agent1.id,
            to_agent_id=agent2.id,
            message="Hello!",
        )
        
        messages = orchestrator.get_messages(agent2.id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello!"
    
    @pytest.mark.asyncio
    async def test_collaborative_task(self):
        orchestrator = MultiAgentOrchestrator()
        
        llm = MockLLM(response="My contribution")
        agent1 = Agent(llm=llm, name="Agent1")
        agent2 = Agent(llm=llm, name="Agent2")
        
        orchestrator.register_agents([agent1, agent2])
        
        results = await orchestrator.execute_collaborative_task(
            task_description="Work together",
            max_rounds=2,
        )
        
        assert "agents_involved" in results
        assert len(results["agents_involved"]) == 2
    
    def test_multi_agent_to_dict(self):
        orchestrator = MultiAgentOrchestrator()
        
        llm = MockLLM()
        agent = Agent(llm=llm, name="TestAgent")
        orchestrator.register_agent(agent)
        
        data = orchestrator.to_dict()
        assert "agents" in data
        assert "TestAgent" in data["agents"]
