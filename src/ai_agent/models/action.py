"""Action 类型体系：整个 Agent Runtime 的统一语言。

设计原则：
1. Planner 只输出 Action
2. Executor 只输入 Action
3. 新增能力通过新增 Action 子类实现，不改核心循环
4. 所有 Action 都是不可变的（Pydantic frozen=True）

Action 类型体系：
- Action（基类）：定义统一接口
- ToolAction：调用工具
- AnswerAction：直接回答用户（结束对话）
- ErrorAction：错误处理

预留扩展点（暂不实现）：
- SkillAction：调用技能
- WorkflowAction：调用工作流
- AgentAction：调用子 Agent
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    """Action 类型枚举。"""

    TOOL = "tool"
    ANSWER = "answer"
    ERROR = "error"


class Action(BaseModel):
    """所有 Action 的基类。

    设计为不可变对象，每次决策生成新实例。
    """

    type: ActionType
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    thought: Optional[str] = None

    model_config = {"frozen": True}

    def is_terminal(self) -> bool:
        """是否是终止性 Action（不需要继续循环）。"""
        return self.type in (ActionType.ANSWER, ActionType.ERROR)


class ToolAction(Action):
    """调用工具的 Action。

    Example:
        ToolAction(
            name="calculator",
            args={"expression": "123 * 456"},
            thought="需要计算 123 × 456 的结果"
        )
    """

    type: ActionType = ActionType.TOOL
    name: str = Field(..., description="工具名称")
    args: Dict[str, Any] = Field(default_factory=dict, description="工具参数")

    @model_validator(mode="after")
    def validate_name(self) -> "ToolAction":
        if not self.name.strip():
            raise ValueError("工具名称不能为空")
        return self


class AnswerAction(Action):
    """直接回答用户的 Action（终止对话）。

    Example:
        AnswerAction(
            content="计算结果是：56088",
            thought="工具已返回结果，可以直接回答用户"
        )
    """

    type: ActionType = ActionType.ANSWER
    content: str = Field(..., description="回答内容")

    @model_validator(mode="after")
    def validate_content(self) -> "AnswerAction":
        if not self.content.strip():
            raise ValueError("回答内容不能为空")
        return self


class ErrorAction(Action):
    """错误处理的 Action。

    Example:
        ErrorAction(
            message="工具调用失败：网络超时",
            thought="工具执行失败，需要告知用户"
        )
    """

    type: ActionType = ActionType.ERROR
    message: str = Field(..., description="错误信息")
    original_action: Optional[Action] = Field(None, description="触发错误的原始 Action")

    @model_validator(mode="after")
    def validate_message(self) -> "ErrorAction":
        if not self.message.strip():
            raise ValueError("错误信息不能为空")
        return self


# ---------- 便捷工厂函数 ----------


def tool_action(
    name: str, args: Optional[Dict[str, Any]] = None, thought: Optional[str] = None
) -> ToolAction:
    """创建 ToolAction 的便捷函数。"""
    return ToolAction(name=name, args=args or {}, thought=thought)


def answer_action(content: str, thought: Optional[str] = None) -> AnswerAction:
    """创建 AnswerAction 的便捷函数。"""
    return AnswerAction(content=content, thought=thought)


def error_action(
    message: str,
    original_action: Optional[Action] = None,
    thought: Optional[str] = None,
) -> ErrorAction:
    """创建 ErrorAction 的便捷函数。"""
    return ErrorAction(
        message=message, original_action=original_action, thought=thought
    )
