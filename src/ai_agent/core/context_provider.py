"""SimpleContextProvider：基于 ConversationStore 的上下文提供者。

从 ConversationStore 获取会话历史，调用会话的知识检索方法，
组装成 AgentContext 供 AgentLoop 使用。

设计原则：
1. 不自己维护状态，完全依赖 ConversationStore
2. 知识检索通过 Conversation.retrieve_knowledge() 内部处理
3. 对外只暴露 ContextProvider 接口
"""

from __future__ import annotations

from typing import List, Optional

from ai_agent.core.conversation import ConversationStore
from ai_agent.core.executor import ContextProvider, ToolProvider
from ai_agent.models.action import Action
from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import AgentContext, MemorySnapshot, RuntimeState


class SimpleContextProvider(ContextProvider):
    """基于 ConversationStore 的上下文提供者。"""

    def __init__(
        self,
        tool_provider: ToolProvider,
        conversation_store: Optional[ConversationStore] = None,
    ) -> None:
        self._tool_provider = tool_provider
        self._store = conversation_store

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def get_context(self, session_id: str, user_input: str) -> AgentContext:
        conversation = await self._get_or_create_conversation(session_id)
        user_msg = ChatMessage(role="user", content=user_input)

        if self._store is not None:
            self._store.append_message(session_id, user_msg)

        knowledge = await conversation.retrieve_knowledge(user_input)

        available_actions = self._tool_provider.as_actions()

        return AgentContext(
            conversation=conversation.messages + [user_msg],
            memory=MemorySnapshot(),
            knowledge=knowledge,
            available_actions=available_actions,
            runtime_state=RuntimeState(
                session_id=session_id,
                iteration=0,
                max_iterations=10,
            ),
            user_input=user_input,
        )

    async def _get_or_create_conversation(self, session_id: str) -> "Conversation":
        if self._store is not None:
            conv = self._store.get(session_id)
            if conv is None:
                conv = self._store.create()
            return conv

        from ai_agent.core.conversation import Conversation

        return Conversation(session_id=session_id)
