"""AgentContext：Agent 决策的完整上下文快照。

设计原则：
1. 不可变：每次循环生成新快照，Planner 只读
2. 完整：包含决策所需的所有信息
3. 解耦：不依赖具体实现，只依赖接口

AgentContext 是整个架构的枢纽：
- Provider 填充数据
- PromptBuilder 消费它生成 Messages
- Planner 消费它做决策

包含的信息：
- conversation: 当前会话的消息列表
- memory: 长期记忆快照
- available_actions: 当前可用的 Action 列表
- runtime_state: 运行时状态（循环次数、超时等）
- user_input: 当前用户输入
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai_agent.models.action import Action
from ai_agent.models.chat import ChatMessage


@dataclass(frozen=True)
class MemorySnapshot:
    """长期记忆的只读快照。

    包含用户的长期信息，如：
    - 用户偏好
    - 历史对话摘要
    - 用户属性（位置、语言、使用的模型等）
    """

    user_profile: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None


@dataclass(frozen=True)
class RuntimeState:
    """Agent 运行时状态。"""

    session_id: str = ""
    iteration: int = 0
    max_iterations: int = 10


@dataclass(frozen=True)
class AgentContext:
    """Agent 决策的完整上下文快照（不可变）。

    每次循环生成新实例，Planner 只读取不修改。

    Example:
        context = AgentContext(
            conversation=conversation,
            memory=MemorySnapshot(user_profile={"name": "Alice"}),
            available_actions=[tool_action("calculator")],
            runtime_state=RuntimeState(session_id="xxx", iteration=1),
            user_input="帮我计算 1+1",
        )
    """

    conversation: List[ChatMessage]
    memory: MemorySnapshot = field(default_factory=MemorySnapshot)
    available_actions: List[Action] = field(default_factory=list)
    runtime_state: RuntimeState = field(default_factory=RuntimeState)
    user_input: str = ""
    system_prompt_snippets: str = ""

    context_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    @property
    def last_message(self) -> Optional[ChatMessage]:
        """获取最后一条消息。"""
        return self.conversation[-1] if self.conversation else None

    @property
    def user_messages(self) -> List[ChatMessage]:
        """获取所有用户消息。"""
        return [m for m in self.conversation if m.role == "user"]

    @property
    def has_memory(self) -> bool:
        """是否有记忆数据。"""
        return (
            self.memory.user_profile is not None
            or self.memory.preferences is not None
            or self.memory.summary is not None
        )
