import asyncio
from typing import Any, Optional

from ai_agent.core.policy import CancellationToken, RetryPolicy
from ai_agent.core.provider import ToolProvider
from ai_agent.models.action import Action, AnswerAction, ErrorAction, ToolAction
from ai_agent.models.context import AgentContext
from ai_agent.models.runtime import (
    Event,
    ExecutionOutcome,
    ExecutionResult,
)
from ai_agent.tools.base import ToolResult


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
                        Event.tool_call(session_id, iteration, action.name, action.args)
                    )
                output = await self._execute_tool(action, token)
                if event_bus is not None:
                    event_bus.emit(
                        Event.tool_result(session_id, iteration, action.name, output)
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
        result = self._execute_once(tool, action.args)
        if result.success:
            return result.output

        last_error_msg = result.error or "未知错误"
        for delay in self._retry_policy.delays():
            token.raise_if_cancelled()
            await asyncio.sleep(delay)
            result = self._execute_once(tool, action.args)
            if result.success:
                return result.output
            last_error_msg = result.error or "未知错误"

        raise RuntimeError(last_error_msg)

    @staticmethod
    def _execute_once(tool, args: dict) -> "ToolResult":
        return tool.execute(args)

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
                            Event.llm_token(
                                session_id=session_id,
                                delta_len=len(delta),
                                delta=delta,
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
