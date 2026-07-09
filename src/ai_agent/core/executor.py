"""ActionExecutor 和工具执行器。

设计原则：
1. Executor 不思考，只执行
2. 工具相关的执行逻辑在此定义
3. Provider 接口已移至 provider.py

层次结构：
- ActionExecutor（总调度）
  - ToolExecutor（执行工具）
  - ...（预留扩展）
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_agent.core.provider import ToolProvider
from ai_agent.models.action import Action, ToolAction


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
