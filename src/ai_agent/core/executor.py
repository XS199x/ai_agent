"""ActionExecutor 和 Provider 接口。

设计原则：
1. Executor 不思考，只执行
2. Provider 只提供资源，不执行
3. 新增能力通过新增 Executor/Provider 实现，不改核心循环

层次结构：
- ActionExecutor（总调度）
  - ToolExecutor（执行工具）
  - ...（预留扩展）

- Provider（资源提供者）
  - ToolProvider（工具资源）
    - LocalToolProvider（本地工具）
    - MCPToolProvider（MCP 工具）
  - ContextProvider（上下文资源）
    - ConversationProvider（会话）
    - MemoryProvider（记忆）
    - KnowledgeProvider（知识）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ai_agent.models.action import Action, ToolAction
from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import (
    AgentContext,
    KnowledgeEntry,
    MemorySnapshot,
    RuntimeState,
)

# ---------- Executor 层 ----------


class ActionExecutor(ABC):
    """Action 执行器接口。"""

    @abstractmethod
    async def execute(self, action: Action) -> str:
        """执行 Action，返回执行结果。"""
        pass


class ToolExecutor(ActionExecutor):
    """工具执行器。

    从 ToolProvider 获取工具并执行。
    """

    def __init__(self, tool_provider: "ToolProvider") -> None:
        self._tool_provider = tool_provider

    async def execute(self, action: Action) -> str:
        if not isinstance(action, ToolAction):
            raise ValueError(f"ToolExecutor 只能执行 ToolAction，收到 {action.type}")

        tool = self._tool_provider.get_tool(action.name)
        if tool is None:
            raise ValueError(f"找不到工具：{action.name}")

        return tool.run(action.args)


# ---------- Provider 层 ----------


class Provider(ABC):
    """所有 Provider 的基类。"""

    @abstractmethod
    async def setup(self) -> None:
        """初始化 Provider。"""
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """清理资源。"""
        pass

    @abstractmethod
    async def health(self) -> bool:
        """健康检查。"""
        pass


class ToolProvider(Provider):
    """工具提供者接口。"""

    @abstractmethod
    def get_tool(self, name: str) -> Optional["BaseTool"]:
        """根据名称获取工具。"""
        pass

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具（包含名称、描述、参数）。"""
        pass

    @abstractmethod
    def as_actions(self) -> List[Action]:
        """把工具列表转换成 Action 列表。"""
        pass


class ContextProvider(Provider):
    """上下文提供者接口。"""

    @abstractmethod
    async def get_context(self, session_id: str, user_input: str) -> AgentContext:
        """构建完整的 AgentContext。"""
        pass


class ConversationProvider(ContextProvider):
    """会话提供者接口。"""

    @abstractmethod
    async def get_conversation(self, session_id: str) -> List[ChatMessage]:
        """获取会话消息列表。"""
        pass

    @abstractmethod
    async def append_message(self, session_id: str, message: ChatMessage) -> None:
        """追加消息到会话。"""
        pass


class MemoryProvider(ContextProvider):
    """记忆提供者接口。"""

    @abstractmethod
    async def get_memory(self, session_id: str) -> MemorySnapshot:
        """获取用户的长期记忆。"""
        pass

    @abstractmethod
    async def update_memory(self, session_id: str, snapshot: MemorySnapshot) -> None:
        """更新用户记忆。"""
        pass


class KnowledgeProvider(ContextProvider):
    """知识提供者接口（RAG）。"""

    @abstractmethod
    async def retrieve(self, query: str, session_id: str) -> List[KnowledgeEntry]:
        """根据查询检索相关知识。"""
        pass


# ---------- 空实现（用于测试和占位） ----------


class EmptyToolProvider(ToolProvider):
    """空工具提供者（没有工具）。"""

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    def get_tool(self, name: str) -> Optional["BaseTool"]:
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        return []

    def as_actions(self) -> List[Action]:
        return []


class EmptyMemoryProvider(MemoryProvider):
    """空记忆提供者（没有记忆）。"""

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def get_context(self, session_id: str, user_input: str) -> AgentContext:
        return AgentContext(
            conversation=[],
            runtime_state=RuntimeState(session_id=session_id),
            user_input=user_input,
        )

    async def get_memory(self, session_id: str) -> MemorySnapshot:
        return MemorySnapshot()

    async def update_memory(self, session_id: str, snapshot: MemorySnapshot) -> None:
        pass


class EmptyKnowledgeProvider(KnowledgeProvider):
    """空知识提供者（没有知识）。"""

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def get_context(self, session_id: str, user_input: str) -> AgentContext:
        return AgentContext(
            conversation=[],
            runtime_state=RuntimeState(session_id=session_id),
            user_input=user_input,
        )

    async def retrieve(self, query: str, session_id: str) -> List[KnowledgeEntry]:
        return []
