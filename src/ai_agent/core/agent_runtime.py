"""AgentRuntime：控制Agent的推理循环和事件发布。"""

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
    """Agent运行时：只负责循环控制、事件发布、上下文流转。"""

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

        async def emit(event: RuntimeEvent) -> None:
            if event_bus is not None:
                event_bus.emit(event)

        try:
            context = await self._context_manager.build_initial(session_id, user_input)
            await emit(RuntimeEvent.started(session_id))

            while True:
                token.raise_if_cancelled()

                policy_result = self._policy.allow_continue(
                    iteration, time.time() - start_time, token
                )
                if not policy_result.allowed:
                    await emit(
                        RuntimeEvent.create_error(
                            session_id,
                            iteration,
                            policy_result.reason or "policy denied",
                        )
                    )
                    return ExecutionResult.from_error(
                        policy_result.reason or "policy denied", ""
                    )

                iteration += 1
                await emit(RuntimeEvent.iteration(session_id, iteration))

                action = await self._planner.plan(context, token)
                await emit(
                    RuntimeEvent.decision(
                        session_id,
                        iteration,
                        action_type=type(action).__name__,
                        thought=action.thought,
                    )
                )

                result = await self._executor.execute(
                    action,
                    context,
                    token,
                    session_id=session_id,
                    iteration=iteration,
                    event_bus=event_bus,
                )

                for event in result.metadata.get("events", []):
                    await emit(event)

                context = await self._context_manager.consume(context, action, result)

                if not result.should_continue:
                    if result.is_success:
                        answer = result.output if isinstance(result.output, str) else ""
                        await emit(
                            RuntimeEvent.done(
                                session_id, iteration, success=True, answer=answer
                            )
                        )
                    else:
                        await emit(
                            RuntimeEvent.done(
                                session_id,
                                iteration,
                                success=False,
                                message=result.error,
                            )
                        )
                    return result

        except asyncio.CancelledError:
            await emit(RuntimeEvent.create_error(session_id, iteration, "cancelled"))
            return ExecutionResult.from_error("cancelled")
        except Exception as e:
            await emit(RuntimeEvent.create_error(session_id, iteration, str(e)))
            return ExecutionResult.from_error(str(e))

    async def run_stream(
        self, session_id: str, user_input: str
    ) -> AsyncGenerator[StreamItem, None]:
        """流式执行：将运行时事件转换为StreamItem流。"""
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
