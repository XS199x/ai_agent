"""细粒度 ContextProvider 实现。

设计原则：
- 每个 Provider 只负责填充 AgentContext 的一个字段
- ContextManager 遍历所有 Provider 收集数据
- 新增能力通过新增 Provider 实现，不改核心循环

包含：
- ConversationProvider：提供对话历史
- MemoryProvider：提供记忆快照
- ApplicationProvider：提供可用动作（应用配置）
- RuntimeProvider：提供运行时状态
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_agent.core.context_manager import ContextProvider
from ai_agent.core.conversation import Conversation, ConversationStore
from ai_agent.core.provider import ToolProvider
from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import MemorySnapshot, RuntimeState


class ConversationProvider(ContextProvider):
    """对话提供者：从 ConversationStore 获取会话历史。"""

    def __init__(self, conversation_store: Optional[ConversationStore] = None) -> None:
        self._store = conversation_store

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        conversation = await self._get_or_create_conversation(session_id)
        user_msg = ChatMessage(role="user", content=user_input)

        if self._store is not None:
            self._store.append_message(session_id, user_msg)

        return {"conversation": conversation.messages + [user_msg]}

    async def _get_or_create_conversation(self, session_id: str) -> "Conversation":
        if self._store is not None:
            conv = self._store.get(session_id)
            if conv is None:
                conv = self._store.create()
            return conv

        from ai_agent.core.conversation import Conversation

        return Conversation(session_id=session_id)


class MemoryProvider(ContextProvider):
    """记忆提供者：提供长期记忆快照。"""

    def __init__(self) -> None:
        pass

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        return {"memory": MemorySnapshot()}


class ApplicationProvider(ContextProvider):
    """应用提供者：提供可用动作列表（应用配置）。"""

    def __init__(self, tool_provider: ToolProvider) -> None:
        self._tool_provider = tool_provider

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        available_actions = self._tool_provider.as_actions()
        return {"available_actions": available_actions}


class RuntimeProvider(ContextProvider):
    """运行时提供者：提供运行时状态（循环次数、超时等）。"""

    def __init__(self, max_iterations: int = 10) -> None:
        self._max_iterations = max_iterations

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        return {
            "runtime_state": RuntimeState(
                session_id=session_id,
                iteration=0,
                max_iterations=self._max_iterations,
            ),
            "user_input": user_input,
        }
