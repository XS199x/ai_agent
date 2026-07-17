import asyncio
from typing import Any, Optional

from ai_agent.core.policy import CancellationToken, RetryPolicy
from ai_agent.core.provider import ToolProvider
from ai_agent.models.action import Action, AnswerAction, ErrorAction, ToolAction
from ai_agent.models.context import AgentContext
from ai_agent.models.runtime import (
    ExecutionOutcome,
    ExecutionResult,
    RuntimeEvent,
    RuntimeEventType,
)


class ActionExecutor:
    def __init__(
        self,
        tool_provider: ToolProvider,
        llm: Any = None,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self._tool_provider = tool_provider
        self._llm = llm
        self._retry_policy = retry_policy or RetryPolicy(max_retries=3)

    async def execute(
        self,
        action: Action,
        context: AgentContext,
        token: Optional[CancellationToken] = None,
        session_id: str = "",
        iteration: int = 0,
        event_bus: Any = None,
    ) -> ExecutionResult:
        token = token or CancellationToken()
        common = {
            "action_trace_id": getattr(action, "trace_id", ""),
        }

        try:
            if isinstance(action, ToolAction):
                token.raise_if_cancelled()
                if event_bus is not None:
                    event_bus.emit(
                        RuntimeEvent.tool_call(
                            session_id, iteration, action.name, action.args
                        )
                    )
                output = await self._execute_tool(action, token)
                if event_bus is not None:
                    event_bus.emit(
                        RuntimeEvent.tool_result(
                            session_id, iteration, action.name, output
                        )
                    )
                return ExecutionResult.success(
                    output, ExecutionOutcome.CONTINUE, **common
                )

            if isinstance(action, AnswerAction):
                token.raise_if_cancelled()
                answer = action.content
                if self._llm is not None:
                    answer = await self._generate_final_answer(
                        context, token, event_bus, session_id
                    )
                return ExecutionResult.success(answer, ExecutionOutcome.STOP, **common)

            if isinstance(action, ErrorAction):
                return ExecutionResult.from_error(
                    action.message, ExecutionOutcome.STOP, **common
                )

            return ExecutionResult.from_error(
                f"不支持的Action类型: {type(action).__name__}",
                ExecutionOutcome.STOP,
                **common,
            )

        except asyncio.CancelledError:
            return ExecutionResult.from_error(
                "cancelled", ExecutionOutcome.STOP, **common
            )
        except Exception as e:
            return ExecutionResult.from_error(str(e), ExecutionOutcome.STOP, **common)

    async def _execute_tool(self, action: ToolAction, token: CancellationToken) -> str:
        tool = self._tool_provider.get_tool(action.name)
        if tool is None:
            raise ValueError(f"找不到工具: {action.name}")

        token.raise_if_cancelled()
        try:
            return tool.run(action.args)
        except Exception as e:
            last_error = e

        for delay in self._retry_policy.delays():
            token.raise_if_cancelled()
            await asyncio.sleep(delay)
            try:
                return tool.run(action.args)
            except Exception as e:
                last_error = e

        raise last_error

    async def _generate_final_answer(
        self,
        context: AgentContext,
        token: CancellationToken,
        event_bus: Any = None,
        session_id: str = "",
    ) -> str:
        from ai_agent.core.message_builder import build_messages
        from ai_agent.prompts.prompt_loader import load_prompt

        token.raise_if_cancelled()
        system_prompt = load_prompt("answer", default="你是一个智能助手。")

        messages = build_messages(context, system_prompt, include_tool_messages=False)

        parts: list[str] = []
        try:
            async for chunk in self._llm.chat_stream(messages=messages):
                token.raise_if_cancelled()
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    continue
                if delta:
                    parts.append(delta)
                    if event_bus is not None:
                        event_bus.emit(
                            RuntimeEvent(
                                type=RuntimeEventType.TOKEN,
                                session_id=session_id,
                                _data={"delta": delta},
                            )
                        )
        except Exception:
            parts = []

        if parts:
            return "".join(parts)

        response = await self._llm.chat(messages=messages)
        try:
            return response.choices[0].message.content or ""
        except Exception:
            return str(response)
