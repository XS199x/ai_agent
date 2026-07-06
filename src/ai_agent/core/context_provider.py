"""SimpleContextProvider：用于测试的简单上下文提供者。

提供基本的上下文构建能力，包含：
- 会话消息
- 可用工具（从 ToolProvider 获取）
- 空的记忆和知识（后续可扩展）
"""

from __future__ import annotations

from typing import List

from ai_agent.core.executor import ContextProvider, ToolProvider
from ai_agent.models.action import Action
from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import AgentContext, MemorySnapshot, RuntimeState


class SimpleContextProvider(ContextProvider):
    """简单的上下文提供者。"""

    def __init__(self, tool_provider: ToolProvider) -> None:
        self._tool_provider = tool_provider
        self._conversations: dict = {}

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def get_context(self, session_id: str, user_input: str) -> AgentContext:
        conversation = self._get_or_create_conversation(session_id)
        conversation.append(ChatMessage(role="user", content=user_input))

        available_actions = self._tool_provider.as_actions()

        return AgentContext(
            conversation=conversation,
            memory=MemorySnapshot(),
            knowledge=[],
            available_actions=available_actions,
            runtime_state=RuntimeState(
                session_id=session_id,
                iteration=0,
                max_iterations=10,
            ),
            user_input=user_input,
        )

    def _get_or_create_conversation(self, session_id: str) -> List[ChatMessage]:
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        return list(self._conversations[session_id])
