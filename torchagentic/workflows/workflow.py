"""
Workflow and Task definitions for TorchAgentic.

This module provides the building blocks for creating multi-step
agent workflows with dependencies and parallel execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import asyncio
import time
import uuid


class TaskStatus(str, Enum):
    """Status of a task in a workflow."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""
    
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def is_success(self) -> bool:
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_failure(self) -> bool:
        return self.status in (TaskStatus.FAILED, TaskStatus.CANCELLED)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class Task:
    """
    Represents a single task in a workflow.
    
    A task can be a function call, an agent execution, or any
    async operation. Tasks can have dependencies on other tasks.
    
    Attributes:
        name: Human-readable name for the task
        func: The function/coroutine to execute
        dependencies: List of task IDs that must complete first
        retry_count: Number of retries on failure
        timeout: Maximum execution time in seconds
        condition: Optional condition function to determine if task should run
    """
    
    def __init__(
        self,
        name: str,
        func: Callable,
        dependencies: Optional[list[str]] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None,
        condition: Optional[Callable[[dict[str, Any]], bool]] = None,
        description: str = "",
        **kwargs,
    ):
        self.id: str = str(uuid.uuid4())
        self.name = name
        self.func = func
        self.dependencies = dependencies or []
        self.retry_count = retry_count
        self.timeout = timeout
        self.condition = condition
        self.description = description
        self.kwargs = kwargs
        
        # Execution state
        self.status: TaskStatus = TaskStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.attempts: int = 0
    
    async def execute(
        self,
        context: dict[str, Any],
        task_results: dict[str, TaskResult],
    ) -> TaskResult:
        """
        Execute the task with the given context.
        
        Args:
            context: Shared context dictionary
            task_results: Results from previously completed tasks
            
        Returns:
            TaskResult with execution outcome
        """
        # Check condition
        if self.condition and not self.condition(context):
            self.status = TaskStatus.SKIPPED
            return TaskResult(
                task_id=self.id,
                status=TaskStatus.SKIPPED,
            )
        
        # Check dependencies
        for dep_id in self.dependencies:
            if dep_id not in task_results:
                return TaskResult(
                    task_id=self.id,
                    status=TaskStatus.FAILED,
                    error=f"Dependency {dep_id} not completed",
                )
            
            dep_result = task_results[dep_id]
            if dep_result.is_failure:
                return TaskResult(
                    task_id=self.id,
                    status=TaskStatus.SKIPPED,
                    error=f"Dependency {dep_id} failed",
                )
        
        # Execute with retries
        last_error = None
        for attempt in range(self.retry_count + 1):
            self.attempts = attempt + 1
            self.status = TaskStatus.RUNNING
            self.started_at = time.time()
            
            try:
                # Prepare arguments from context and dependency results
                args = self._prepare_args(context, task_results)
                
                # Execute with timeout
                if asyncio.iscoroutinefunction(self.func):
                    if self.timeout:
                        result = await asyncio.wait_for(
                            self.func(**args),
                            timeout=self.timeout,
                        )
                    else:
                        result = await self.func(**args)
                else:
                    result = self.func(**args)
                
                self.result = result
                self.status = TaskStatus.COMPLETED
                self.completed_at = time.time()
                
                return TaskResult(
                    task_id=self.id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    execution_time=self.completed_at - self.started_at,
                    started_at=self.started_at,
                    completed_at=self.completed_at,
                )
                
            except asyncio.TimeoutError:
                last_error = f"Task timed out after {self.timeout}s"
            except Exception as e:
                last_error = str(e)
        
        # All retries exhausted
        self.status = TaskStatus.FAILED
        self.error = last_error
        self.completed_at = time.time()
        
        return TaskResult(
            task_id=self.id,
            status=TaskStatus.FAILED,
            error=last_error,
            execution_time=self.completed_at - self.started_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )
    
    def _prepare_args(
        self,
        context: dict[str, Any],
        task_results: dict[str, TaskResult],
    ) -> dict[str, Any]:
        """Prepare arguments for the task function."""
        args = dict(self.kwargs)
        
        # Add context
        args["context"] = context
        
        # Add dependency results
        for dep_id in self.dependencies:
            if dep_id in task_results:
                args[f"dep_{dep_id[:8]}"] = task_results[dep_id].result
        
        return args
    
    def reset(self) -> None:
        """Reset task state for re-execution."""
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.attempts = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "result": self.result,
            "error": self.error,
        }
    
    def __repr__(self) -> str:
        return f"Task(name={self.name}, status={self.status.value})"


class Workflow:
    """
    A collection of tasks with dependencies that form a workflow.
    
    Workflows can be executed sequentially or in parallel based on
    task dependencies. They track execution state and results.
    
    Attributes:
        name: Name of the workflow
        description: Optional description
        tasks: Dictionary of tasks by ID
        context: Shared context for all tasks
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
    ):
        self.id: str = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tasks: dict[str, Task] = {}
        self.context: dict[str, Any] = {}
        
        # Execution state
        self.status: TaskStatus = TaskStatus.PENDING
        self.results: dict[str, TaskResult] = {}
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
    
    def add_task(
        self,
        name: str,
        func: Callable,
        dependencies: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """
        Add a task to the workflow.
        
        Args:
            name: Task name
            func: Function to execute
            dependencies: List of task names this task depends on
            **kwargs: Arguments to pass to the function
            
        Returns:
            Task ID
        """
        # Resolve dependency names to IDs
        dep_ids = []
        if dependencies:
            for dep_name in dependencies:
                dep_task = self.get_task_by_name(dep_name)
                if dep_task:
                    dep_ids.append(dep_task.id)
                else:
                    raise ValueError(f"Dependency task '{dep_name}' not found")
        
        task = Task(
            name=name,
            func=func,
            dependencies=dep_ids,
            **kwargs,
        )
        self.tasks[task.id] = task
        
        return task.id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_task_by_name(self, name: str) -> Optional[Task]:
        """Get a task by name."""
        for task in self.tasks.values():
            if task.name == name:
                return task
        return None
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a value in the shared context."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared context."""
        return self.context.get(key, default)
    
    def _get_ready_tasks(self) -> list[Task]:
        """Get tasks that are ready to run (all dependencies satisfied)."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_satisfied = True
            for dep_id in task.dependencies:
                if dep_id not in self.results:
                    deps_satisfied = False
                    break
                if self.results[dep_id].status != TaskStatus.COMPLETED:
                    deps_satisfied = False
                    break
            
            if deps_satisfied:
                ready.append(task)
        
        return ready
    
    async def execute(
        self,
        context: Optional[dict[str, Any]] = None,
        parallel: bool = True,
    ) -> dict[str, TaskResult]:
        """
        Execute the workflow.
        
        Args:
            context: Optional initial context
            parallel: Whether to run independent tasks in parallel
            
        Returns:
            Dictionary of task results
        """
        if context:
            self.context.update(context)
        
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
        self.results = {}
        
        # Reset all tasks
        for task in self.tasks.values():
            task.reset()
        
        while True:
            ready_tasks = self._get_ready_tasks()
            
            if not ready_tasks:
                # Check if we're done or stuck
                pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
                if not pending:
                    break
                
                # If there are pending tasks but none ready, we have a problem
                if ready_tasks:
                    break
                raise ValueError("Workflow has circular dependencies or failed prerequisites")
            
            if parallel:
                # Run all ready tasks in parallel
                results = await asyncio.gather(
                    *[
                        task.execute(self.context, self.results)
                        for task in ready_tasks
                    ],
                    return_exceptions=True,
                )
                
                for task, result in zip(ready_tasks, results):
                    if isinstance(result, Exception):
                        task.status = TaskStatus.FAILED
                        task.error = str(result)
                        task.completed_at = time.time()
                        result = TaskResult(
                            task_id=task.id,
                            status=TaskStatus.FAILED,
                            error=str(result),
                        )
                    self.results[task.id] = result
            else:
                # Run tasks sequentially
                for task in ready_tasks:
                    result = await task.execute(self.context, self.results)
                    self.results[task.id] = result
        
        self.completed_at = time.time()
        
        # Determine overall status
        failed = [r for r in self.results.values() if r.is_failure]
        self.status = TaskStatus.FAILED if failed else TaskStatus.COMPLETED
        
        return self.results
    
    def get_result(self, task_name: str) -> Optional[Any]:
        """Get the result of a task by name."""
        task = self.get_task_by_name(task_name)
        if task and task.id in self.results:
            return self.results[task.id].result
        return None
    
    def reset(self) -> None:
        """Reset workflow state for re-execution."""
        self.status = TaskStatus.PENDING
        self.results = {}
        self.started_at = None
        self.completed_at = None
        
        for task in self.tasks.values():
            task.reset()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert workflow to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "results": {tid: result.to_dict() for tid, result in self.results.items()},
            "context": self.context,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
    
    def visualize(self) -> str:
        """Create a text visualization of the workflow."""
        lines = [f"Workflow: {self.name}", "=" * 40]
        
        for task in self.tasks.values():
            deps = []
            for dep_id in task.dependencies:
                dep_task = self.get_task(dep_id)
                if dep_task:
                    deps.append(dep_task.name)
            
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.RUNNING: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.SKIPPED: "⏭️",
            }.get(task.status, "❓")
            
            deps_str = f" <- [{', '.join(deps)}]" if deps else ""
            lines.append(f"{status_icon} {task.name}{deps_str}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Workflow(name={self.name}, tasks={len(self.tasks)}, status={self.status.value})"
