import json
import logging
from typing import Any, Dict, List, Optional, Protocol

from ai_agent.core.event import Event
from ai_agent.models.action import Action, AnswerAction, ErrorAction, ToolAction
from ai_agent.models.chat import ChatMessage, FunctionCall, ToolCall
from ai_agent.models.context import AgentContext, MemorySnapshot, RuntimeState
from ai_agent.models.runtime import ExecutionResult

_logger = logging.getLogger(__name__)


class ContextProvider(Protocol):
    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]: ...


class ContextManager:
    def __init__(
        self,
        providers: List[ContextProvider],
        bus: Optional[Any] = None,
    ) -> None:
        self._providers = providers
        self._bus = bus

    async def build_initial(self, session_id: str, user_input: str) -> AgentContext:
        data: Dict[str, Any] = {
            "conversation": [],
            "memory": MemorySnapshot(),
            "available_actions": [],
            "runtime_state": RuntimeState(session_id=session_id),
            "user_input": user_input,
        }

        for provider in self._providers:
            try:
                data.update(await provider.provide(session_id, user_input))
            except Exception as e:
                _logger.warning("Provider %s failed: %s", provider, e)

        return AgentContext(
            conversation=data.get("conversation", []),
            memory=data.get("memory", MemorySnapshot()),
            available_actions=data.get("available_actions", []),
            runtime_state=data.get(
                "runtime_state", RuntimeState(session_id=session_id)
            ),
            user_input=data.get("user_input", user_input),
            system_prompt_snippets=data.get("system_prompt_snippets", ""),
        )

    async def consume(
        self, context: AgentContext, action: Action, result: ExecutionResult
    ) -> AgentContext:
        messages_to_add: List[ChatMessage] = []

        if isinstance(action, ToolAction):
            tool_call_id = action.trace_id
            messages_to_add.append(
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=tool_call_id,
                            type="function",
                            function=FunctionCall(
                                name=action.name,
                                arguments=json.dumps(action.args),
                            ),
                        )
                    ],
                )
            )
            if result.output is not None:
                messages_to_add.append(
                    ChatMessage(
                        role="tool",
                        content=str(result.output),
                        tool_call_id=tool_call_id,
                    )
                )

        elif isinstance(action, AnswerAction):
            messages_to_add.append(
                ChatMessage(
                    role="assistant",
                    content=str(result.output or action.content),
                )
            )

        elif isinstance(action, ErrorAction):
            messages_to_add.append(
                ChatMessage(
                    role="assistant",
                    content=f"错误：{action.message}",
                )
            )

        if self._bus is not None:
            for msg in messages_to_add:
                self._bus.emit(
                    Event(
                        name="conversation.append",
                        payload={
                            "session_id": context.runtime_state.session_id,
                            "message": msg,
                        },
                    )
                )

        return AgentContext(
            conversation=list(context.conversation) + messages_to_add,
            memory=context.memory,
            available_actions=context.available_actions,
            runtime_state=RuntimeState(
                session_id=context.runtime_state.session_id,
                iteration=context.runtime_state.iteration + 1,
                max_iterations=context.runtime_state.max_iterations,
            ),
            user_input=context.user_input,
            system_prompt_snippets=context.system_prompt_snippets,
        )
