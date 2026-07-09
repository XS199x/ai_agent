"""ContextManager：上下文管理器。

职责：
1. 管理多个 ContextProvider
2. 遍历所有 Provider 收集数据
3. 组装成完整的 AgentContext
4. 更新上下文（工具执行结果回灌）

设计原则：
- ContextManager 不自己生成数据，只负责组装
- 每个 Provider 只关心自己的字段
- 支持动态添加/移除 Provider

数据流：
  Provider.provide() → ContextManager.build_initial() → AgentContext
                          ↓
                    Planner.plan() → Action
                          ↓
                    ActionDispatcher.dispatch() → Result
                          ↓
                    ContextManager.update() → 新 AgentContext
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ai_agent.core.provider import Provider
from ai_agent.models.action import Action, AnswerAction, ToolAction
from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import AgentContext, MemorySnapshot, RuntimeState


class ContextProvider(Provider):
    """上下文提供者接口：填充 AgentContext 的某个局部。

    设计原则：
    - 每个 Provider 只负责填充 AgentContext 的一个或多个字段
    - ContextManager 遍历所有 Provider，收集数据后组装成完整的 AgentContext
    """

    @abstractmethod
    async def provide(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """提供上下文数据片段。

        返回一个字典，键为 AgentContext 的字段名，值为对应的数据。
        例如：{"conversation": [...], "knowledge": [...]}
        """
        pass


class ContextManager(ABC):
    """上下文管理器接口。"""

    @abstractmethod
    async def build_initial(self, session_id: str, user_input: str) -> AgentContext:
        """构建初始上下文。"""
        pass

    @abstractmethod
    async def update(
        self, context: AgentContext, action: Action, observation: Any
    ) -> AgentContext:
        """根据动作和观察更新上下文。"""
        pass

    @abstractmethod
    def build_llm_messages(self, context: AgentContext) -> List[Dict[str, Any]]:
        """将 AgentContext 转换为 LLM 消息格式。"""
        pass

    @abstractmethod
    def summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """生成消息摘要，用于日志和调试。"""
        pass


class DefaultContextManager(ContextManager):
    """默认上下文管理器：遍历所有 Provider 收集数据。"""

    def __init__(
        self,
        providers: List[ContextProvider],
        llm: Optional[Any] = None,
        answer_prompt: str = "",
    ) -> None:
        self._providers = providers
        self._llm = llm
        self._answer_prompt = answer_prompt

    async def build_initial(self, session_id: str, user_input: str) -> AgentContext:
        """遍历所有 Provider 收集数据，组装成完整的 AgentContext。"""
        data: Dict[str, Any] = {
            "conversation": [],
            "memory": MemorySnapshot(),
            "available_actions": [],
            "runtime_state": RuntimeState(session_id=session_id),
            "user_input": user_input,
        }

        for provider in self._providers:
            try:
                provider_data = await provider.provide(session_id, user_input)
                data.update(provider_data)
            except Exception as e:
                pass

        return AgentContext(
            conversation=data.get("conversation", []),
            memory=data.get("memory", MemorySnapshot()),
            available_actions=data.get("available_actions", []),
            runtime_state=data.get(
                "runtime_state", RuntimeState(session_id=session_id)
            ),
            user_input=data.get("user_input", user_input),
        )

    async def update(
        self, context: AgentContext, action: Action, observation: Any
    ) -> AgentContext:
        """将工具执行结果追加到对话历史，生成新的上下文。"""
        if isinstance(action, ToolAction):
            content = f"工具调用: {action.name}({action.args})\n结果: {observation}"
        else:
            content = f"动作: {action.type.value}\n结果: {observation}"

        tool_msg = ChatMessage(
            role="assistant",
            content=content,
        )

        new_conversation = list(context.conversation) + [tool_msg]

        return AgentContext(
            conversation=new_conversation,
            memory=context.memory,
            available_actions=context.available_actions,
            runtime_state=RuntimeState(
                session_id=context.runtime_state.session_id,
                iteration=context.runtime_state.iteration + 1,
                max_iterations=context.runtime_state.max_iterations,
            ),
            user_input=context.user_input,
        )

    def build_llm_messages(self, context: AgentContext) -> List[ChatMessage]:
        """将 AgentContext 转换为 LLM 消息格式。"""
        messages: List[ChatMessage] = []

        if self._answer_prompt:
            messages.append(ChatMessage(role="system", content=self._answer_prompt))

        messages.extend(context.conversation)

        return messages

    def summarize_messages(self, messages: List[ChatMessage]) -> str:
        """生成消息摘要。"""
        role_counts: Dict[str, int] = {}
        total_chars = 0

        for msg in messages:
            role = msg.role
            role_counts[role] = role_counts.get(role, 0) + 1
            total_chars += len(str(msg.content))

        parts = [f"{k}:{v}" for k, v in role_counts.items()]
        parts.append(f"总字符:{total_chars}")

        return ", ".join(parts)
