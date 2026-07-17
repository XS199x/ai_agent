"""AgentRuntime: Control loop and event emission."""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator, Optional

from ai_agent.core.context_manager import ContextManager
from ai_agent.core.event import EventBus
from ai_agent.core.planner import Planner
from ai_agent.core.policy import CancellationToken, RuntimePolicy
from ai_agent.core.stream import StreamHandle, StreamItem
from ai_agent.models.context import AgentContext
from ai_agent.models.runtime import ExecutionResult, RuntimeEvent


class AgentRuntime:
    """Agent runtime: only responsible for loop control, event dispatch, context flow."""

    def __init__(
        self,
        planner: Planner,
        context_manager: ContextManager,
        executor: Any,
        bus: Optional[EventBus] = None,
        policy: Optional[RuntimePolicy] = None,
    ) -> None:
        self._planner = planner
        self._context_manager = context_manager
        self._executor = executor
        self._bus = bus
        self._policy = policy or RuntimePolicy()

    @staticmethod
    def _emit(event_bus: Optional[EventBus], event: RuntimeEvent) -> None:
        if event_bus is not None:
            event_bus.emit(event)

    async def run(
        self,
        session_id: str,
        user_input: str,
        token: Optional[CancellationToken] = None,
        bus: Optional[EventBus] = None,
    ) -> ExecutionResult:
        token = token or CancellationToken()
        event_bus = bus or self._bus
        start_time = time.time()
        iteration = 0

        try:
            context = await self._context_manager.build_initial(session_id, user_input)
            self._emit(event_bus, RuntimeEvent.started(session_id))

            while True:
                token.raise_if_cancelled()

                policy_result = self._policy.allow_continue(
                    iteration, time.time() - start_time, token
                )
                if not policy_result.allowed:
                    self._emit(
                        event_bus,
                        RuntimeEvent.create_error(
                            session_id,
                            iteration,
                            policy_result.reason or "policy denied",
                        ),
                    )
                    return ExecutionResult.from_error(
                        policy_result.reason or "policy denied", ""
                    )

                iteration += 1
                self._emit(
                    event_bus, RuntimeEvent.iteration_event(session_id, iteration)
                )

                action = await self._planner.plan(context, token)
                self._emit(
                    event_bus,
                    RuntimeEvent.decision(
                        session_id,
                        iteration,
                        action_type=type(action).__name__,
                        thought=action.thought,
                    ),
                )

                result = await self._executor.execute(
                    action,
                    context,
                    token,
                    session_id=session_id,
                    iteration=iteration,
                    event_bus=event_bus,
                )

                context = await self._context_manager.consume(context, action, result)

                if not result.should_continue:
                    if result.is_success:
                        answer = result.output if isinstance(result.output, str) else ""
                        self._emit(
                            event_bus,
                            RuntimeEvent.done(
                                session_id, iteration, success=True, answer=answer
                            ),
                        )
                    else:
                        self._emit(
                            event_bus,
                            RuntimeEvent.create_error(
                                session_id, iteration, result.error or "unknown error"
                            ),
                        )
                    return result

        except asyncio.CancelledError:
            self._emit(
                event_bus, RuntimeEvent.create_error(session_id, iteration, "cancelled")
            )
            return ExecutionResult.from_error("cancelled")
        except Exception as e:
            self._emit(
                event_bus, RuntimeEvent.create_error(session_id, iteration, str(e))
            )
            return ExecutionResult.from_error(str(e))

    async def run_stream(
        self, session_id: str, user_input: str
    ) -> AsyncGenerator[StreamItem, None]:
        """Stream execution: convert runtime events to StreamItem flow."""
        from ai_agent.core.observer import StreamEventObserver

        session_bus = EventBus()
        handle = StreamHandle(bus=self._bus, session_id=session_id)
        session_bus.subscribe(StreamEventObserver(handle))

        token = CancellationToken()
        token.add_listener(
            lambda reason: handle.emit_event("cancelled", {"reason": reason})
        )

        task = asyncio.create_task(
            self.run(session_id, user_input, token, bus=session_bus)
        )

        try:
            async for item in handle.stream():
                yield item
        finally:
            token.cancel("stream_closed")
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if task.done() and task.exception() is not None and not handle.done:
                handle.emit_error(str(task.exception()))
