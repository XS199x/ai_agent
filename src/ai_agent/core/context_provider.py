from __future__ import annotations

from typing import Any, Dict, Optional

from ai_agent.core.context_manager import ContextProvider
from ai_agent.core.provider import ToolProvider
from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import MemorySnapshot, RuntimeState
from ai_agent.models.runtime import Event
from ai_agent.persistence.models import Conversation
from ai_agent.persistence.store import ConversationStore


class ConversationProvider:
    def __init__(
        self,
        conversation_store: Optional[ConversationStore] = None,
        bus: Optional[Any] = None,
    ) -> None:
        self._store = conversation_store
        self._bus = bus

    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        conversation = await self._get_or_create_conversation(session_id)
        user_msg = ChatMessage(role="user", content=user_input)

        if self._bus is not None:
            self._bus.emit(
                Event(
                    name="conversation.append",
                    payload={"session_id": session_id, "message": user_msg},
                )
            )

        return {"conversation": conversation.messages + [user_msg]}

    async def _get_or_create_conversation(self, session_id: str) -> Conversation:
        if self._store is not None:
            conv = self._store.get(session_id)
            if conv is None:
                conv = self._store.create()
            return conv
        return Conversation(session_id=session_id)


class MemoryProvider:
    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        return {"memory": MemorySnapshot()}


class ApplicationProvider:
    def __init__(
        self,
        tool_provider: ToolProvider,
        extra_prompt_snippets: Any = None,
    ) -> None:
        self._tool_provider = tool_provider
        self._extra_snippets = extra_prompt_snippets

    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        available_actions = self._tool_provider.as_actions()
        snippets = ""
        if self._extra_snippets is not None:
            try:
                snippets = str(self._extra_snippets() or "")
            except Exception:
                snippets = ""
        return {
            "available_actions": available_actions,
            "system_prompt_snippets": snippets,
        }


class RuntimeProvider:
    def __init__(self, max_iterations: int = 10) -> None:
        self._max_iterations = max_iterations

    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        return {
            "runtime_state": RuntimeState(
                session_id=session_id,
                iteration=0,
                max_iterations=self._max_iterations,
            ),
            "user_input": user_input,
        }
