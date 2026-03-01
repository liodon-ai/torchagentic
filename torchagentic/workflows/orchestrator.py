"""
Orchestrator for managing workflows and multi-agent systems.
"""

from typing import Any, Optional
import asyncio
import time
import uuid

from torchagentic.core.agent import Agent
from torchagentic.core.response import AgentResponse
from torchagentic.workflows.workflow import Workflow, Task, TaskStatus, TaskResult


class Orchestrator:
    """
    Orchestrates workflow execution and manages shared state.
    
    The orchestrator is responsible for:
    - Managing workflow lifecycle
    - Handling errors and retries
    - Coordinating task execution
    - Providing observability
    
    Attributes:
        workflows: Dictionary of registered workflows
        max_parallel: Maximum parallel workflow executions
    """
    
    def __init__(self, max_parallel: int = 5):
        self.id: str = str(uuid.uuid4())
        self.workflows: dict[str, Workflow] = {}
        self.max_parallel = max_parallel
        
        # Execution history
        self._history: list[dict[str, Any]] = []
        self._active_executions: dict[str, dict[str, Any]] = {}
    
    def register_workflow(self, workflow: Workflow) -> str:
        """Register a workflow."""
        self.workflows[workflow.id] = workflow
        return workflow.id
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def get_workflow_by_name(self, name: str) -> Optional[Workflow]:
        """Get a workflow by name."""
        for workflow in self.workflows.values():
            if workflow.name == name:
                return workflow
        return None
    
    def create_workflow(self, name: str, description: str = "") -> Workflow:
        """Create and register a new workflow."""
        workflow = Workflow(name=name, description=description)
        self.register_workflow(workflow)
        return workflow
    
    async def execute_workflow(
        self,
        workflow_id: str,
        context: Optional[dict[str, Any]] = None,
        parallel: bool = True,
    ) -> dict[str, TaskResult]:
        """
        Execute a workflow.
        
        Args:
            workflow_id: ID of workflow to execute
            context: Initial context
            parallel: Whether to run tasks in parallel
            
        Returns:
            Dictionary of task results
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        # Track execution
        exec_id = str(uuid.uuid4())
        self._active_executions[exec_id] = {
            "workflow_id": workflow_id,
            "workflow_name": workflow.name,
            "started_at": time.time(),
            "context": context or {},
        }
        
        try:
            results = await workflow.execute(context=context, parallel=parallel)
            
            # Record in history
            self._history.append({
                "exec_id": exec_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "status": workflow.status.value,
                "started_at": workflow.started_at,
                "completed_at": workflow.completed_at,
                "task_count": len(workflow.tasks),
                "success_count": len([r for r in results.values() if r.is_success]),
                "failure_count": len([r for r in results.values() if r.is_failure]),
            })
            
            return results
        
        finally:
            self._active_executions.pop(exec_id, None)
    
    async def execute_workflow_by_name(
        self,
        name: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> dict[str, TaskResult]:
        """Execute a workflow by name."""
        workflow = self.get_workflow_by_name(name)
        if not workflow:
            raise ValueError(f"Workflow not found: {name}")
        return await self.execute_workflow(workflow.id, context=context, **kwargs)
    
    def get_active_executions(self) -> list[dict[str, Any]]:
        """Get currently active executions."""
        return list(self._active_executions.values())
    
    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get execution history."""
        return self._history[-limit:]
    
    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        total_executions = len(self._history)
        successful = len([h for h in self._history if h["status"] == "completed"])
        
        return {
            "total_workflows": len(self.workflows),
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": total_executions - successful,
            "success_rate": successful / max(1, total_executions),
            "active_executions": len(self._active_executions),
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert orchestrator to dictionary format."""
        return {
            "id": self.id,
            "workflows": {
                wid: {"name": w.name, "status": w.status.value, "tasks": len(w.tasks)}
                for wid, w in self.workflows.items()
            },
            "stats": self.get_stats(),
            "active_executions": len(self._active_executions),
        }
    
    def __repr__(self) -> str:
        return f"Orchestrator(workflows={len(self.workflows)})"


class MultiAgentOrchestrator(Orchestrator):
    """
    Orchestrator specialized for multi-agent workflows.
    
    Extends the base orchestrator with agent-specific capabilities:
    - Agent registration and management
    - Agent-to-agent communication
    - Collaborative task execution
    """
    
    def __init__(self, max_parallel: int = 5):
        super().__init__(max_parallel=max_parallel)
        self.agents: dict[str, Agent] = {}
        self._agent_messages: dict[str, list[dict[str, Any]]] = {}
    
    def register_agent(self, agent: Agent) -> str:
        """Register an agent with the orchestrator."""
        self.agents[agent.id] = agent
        self._agent_messages[agent.id] = []
        return agent.id
    
    def register_agents(self, agents: list[Agent]) -> list[str]:
        """Register multiple agents."""
        return [self.register_agent(agent) for agent in agents]
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        for agent in self.agents.values():
            if agent.name == name:
                return agent
        return None
    
    async def send_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        message: str,
    ) -> None:
        """Send a message from one agent to another."""
        if to_agent_id not in self._agent_messages:
            self._agent_messages[to_agent_id] = []
        
        self._agent_messages[to_agent_id].append({
            "from": from_agent_id,
            "content": message,
            "timestamp": time.time(),
        })
    
    def get_messages(self, agent_id: str, clear: bool = True) -> list[dict[str, Any]]:
        """Get pending messages for an agent."""
        messages = self._agent_messages.get(agent_id, [])
        if clear:
            self._agent_messages[agent_id] = []
        return messages
    
    async def execute_collaborative_task(
        self,
        task_description: str,
        agent_ids: Optional[list[str]] = None,
        max_rounds: int = 5,
    ) -> dict[str, Any]:
        """
        Execute a task collaboratively across multiple agents.
        
        Args:
            task_description: Description of the task
            agent_ids: List of agent IDs to involve (all if None)
            max_rounds: Maximum conversation rounds
            
        Returns:
            Combined results from all agents
        """
        if agent_ids is None:
            agent_ids = list(self.agents.keys())
        
        agents = [self.agents[aid] for aid in agent_ids if aid in self.agents]
        if not agents:
            return {"error": "No agents available"}
        
        results = {}
        
        # Round-robin execution
        for round_num in range(max_rounds):
            for agent in agents:
                # Get any messages for this agent
                messages = self.get_messages(agent.id)
                
                # Build prompt
                prompt = task_description
                if messages:
                    prompt += "\n\nMessages from other agents:\n"
                    for msg in messages:
                        from_agent = self.get_agent(msg["from"])
                        from_name = from_agent.name if from_agent else msg["from"]
                        prompt += f"- {from_name}: {msg['content']}\n"
                
                # Execute agent
                response = await agent.act(prompt)
                
                # Store result
                results[f"{agent.name}_round_{round_num}"] = response.to_dict()
                
                # Share response with other agents
                if response.content:
                    for other_agent in agents:
                        if other_agent.id != agent.id:
                            await self.send_message(
                                agent.id,
                                other_agent.id,
                                response.content,
                            )
        
        return {
            "task_description": task_description,
            "agents_involved": [a.name for a in agents],
            "rounds_completed": max_rounds,
            "results": results,
        }
    
    async def create_agent_workflow(
        self,
        name: str,
        agent_tasks: list[dict[str, Any]],
    ) -> Workflow:
        """
        Create a workflow from agent task specifications.
        
        Args:
            name: Workflow name
            agent_tasks: List of task specs with agent assignments
            
        Returns:
            Created workflow
        """
        workflow = self.create_workflow(name=name)
        
        prev_task_id = None
        for task_spec in agent_tasks:
            agent_name = task_spec.get("agent")
            task_name = task_spec.get("name", f"task_{uuid.uuid4().hex[:8]}")
            task_input = task_spec.get("input", "")
            
            # Find agent
            agent = self.get_agent_by_name(agent_name)
            if not agent:
                raise ValueError(f"Agent not found: {agent_name}")
            
            # Create task
            async def agent_task(context, agent=agent, input_text=task_input):
                # Substitute context variables in input
                for key, value in context.items():
                    input_text = input_text.replace(f"{{{key}}}", str(value))
                
                response = await agent.act(input_text)
                return response
            
            dependencies = [prev_task_id] if prev_task_id else None
            prev_task_id = workflow.add_task(
                name=task_name,
                func=agent_task,
                dependencies=dependencies,
            )
        
        return workflow
    
    def get_agent_stats(self) -> dict[str, Any]:
        """Get statistics about registered agents."""
        return {
            "total_agents": len(self.agents),
            "agents": {
                agent.name: agent.get_state()
                for agent in self.agents.values()
            },
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        base_dict = super().to_dict()
        base_dict["agents"] = {
            agent.name: {"id": agent.id, "type": agent.__class__.__name__}
            for agent in self.agents.values()
        }
        return base_dict
    
    def __repr__(self) -> str:
        return (
            f"MultiAgentOrchestrator("
            f"workflows={len(self.workflows)}, "
            f"agents={len(self.agents)})"
        )
