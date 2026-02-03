from abc import ABC, abstractmethod
import json
from typing import AsyncIterator, Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE

    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> str:
        """Execute a single step: think and act."""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()

    async def run_stream(self, request: Optional[str] = None) -> AsyncIterator[str]:
        """Execute the agent's main loop, yielding step results as they occur.

        Overrides BaseAgent.run_stream to provide more granular updates (think vs act).
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")

                # THINK
                should_act = await self.think()

                # Yield thought if available
                if (
                    self.memory.messages
                    and self.memory.messages[-1].role == "assistant"
                ):
                    thought_content = self.memory.messages[-1].content
                    if thought_content:
                        yield f"Thought: {thought_content}"

                if not should_act:
                    yield "Thinking complete - no action needed"
                    break

                # ACT
                # Yield action execution status
                # (Optional: In the future, we could yield individual tool calls here if available in memory)

                step_result = await self.act()

                tool_events = getattr(self, "_last_tool_events", None) or []
                browser_snapshot_emitted = False
                for event in tool_events:
                    if event.get("name") == "browser_use":
                        args = event.get("arguments") or {}
                        action = args.get("action")
                        action_payload = {
                            "action": action,
                            "args": args,
                        }
                        yield f"BrowserAction: {json.dumps(action_payload, ensure_ascii=False)}"
                        if event.get("base64_image"):
                            browser_snapshot_emitted = True
                            yield f"BrowserSnapshot: {event['base64_image']}"

                # Yield snapshot if available from tool execution
                if (
                    hasattr(self, "_current_base64_image")
                    and self._current_base64_image
                    and not browser_snapshot_emitted
                ):
                    yield f"Snapshot: {self._current_base64_image}"

                # Yield observation/result
                yield f"Observation: {step_result}"

                # Check for stuck state
                if self.is_stuck():
                    self.handle_stuck_state()

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                yield f"Terminated: Reached max steps ({self.max_steps})"

        await SANDBOX_CLIENT.cleanup()
