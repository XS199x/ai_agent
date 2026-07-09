"""ActionDispatcher：动作分发器。

职责：
1. 注册 Action Handler
2. 根据 Action 类型分发到对应 Handler
3. 执行 Action 并返回结果

设计原则：
- 无业务逻辑，只做分发
- 新增 Action 类型只需注册新 Handler，不改核心代码
- Handler 模式解耦 Action 类型与执行逻辑
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from ai_agent.models.action import Action, AnswerAction, ErrorAction, ToolAction


class ActionHandler(ABC):
    """Action Handler 接口。"""

    @abstractmethod
    def can_handle(self, action: Action) -> bool:
        """判断是否能处理该 Action。"""
        pass

    @abstractmethod
    async def execute(self, action: Action) -> Any:
        """执行 Action，返回执行结果。"""
        pass


class ToolActionHandler(ActionHandler):
    """工具动作 Handler。"""

    def __init__(self, tool_executor: Any) -> None:
        self._tool_executor = tool_executor

    def can_handle(self, action: Action) -> bool:
        return isinstance(action, ToolAction)

    async def execute(self, action: Action) -> Any:
        from ai_agent.core.executor import ActionExecutor

        if isinstance(self._tool_executor, ActionExecutor):
            return await self._tool_executor.execute(action)
        return await self._tool_executor.execute(action)


class AnswerActionHandler(ActionHandler):
    """回答动作 Handler。"""

    def can_handle(self, action: Action) -> bool:
        return isinstance(action, AnswerAction)

    async def execute(self, action: Action) -> Any:
        if isinstance(action, AnswerAction):
            return action.content
        return ""


class ErrorActionHandler(ActionHandler):
    """错误动作 Handler。"""

    def can_handle(self, action: Action) -> bool:
        return isinstance(action, ErrorAction)

    async def execute(self, action: Action) -> Any:
        if isinstance(action, ErrorAction):
            raise RuntimeError(action.message)
        raise RuntimeError("未知错误")


class ActionDispatcher:
    """动作分发器。

    使用注册模式管理 Handler，根据 Action 类型自动选择 Handler。
    """

    _action_types = [ToolAction, AnswerAction, ErrorAction]

    def __init__(self) -> None:
        self._handlers: Dict[Type[Action], ActionHandler] = {}

    def register_handler(self, handler: ActionHandler) -> None:
        """注册 Action Handler。

        根据 Handler 的 can_handle 方法自动推断对应的 Action 类型。
        """
        for action_type in self._action_types:
            test_action = action_type.__new__(action_type)
            if handler.can_handle(test_action):
                self._handlers[action_type] = handler
                return
        raise ValueError(
            f"无法推断 Handler 对应的 Action 类型：{type(handler).__name__}"
        )

    def get_handler(self, action: Action) -> Optional[ActionHandler]:
        """获取 Action 对应的 Handler。"""
        action_type = type(action)
        return self._handlers.get(action_type)

    async def dispatch(self, action: Action) -> Any:
        """分发 Action 到对应 Handler 执行。"""
        handler = self.get_handler(action)
        if handler is None:
            raise ValueError(f"找不到 Action Handler：{type(action).__name__}")
        return await handler.execute(action)
