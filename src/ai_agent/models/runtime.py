"""Runtime 核心模型：定义执行结果和运行时事件。

设计原则：
1. ExecutionResult：统一的执行结果封装，包含成功/失败、输出、是否继续
2. RuntimeEvent：继承 Event，用于 Agent 运行时事件，有强类型元数据
3. 所有模型不可变，确保线程安全

ExecutionResult：
- 统一返回格式，Runtime 只关心 result.should_continue
- 异常由 Executor 包装，Runtime 不需要 try-catch

RuntimeEvent：
- 继承 Event，共享 name、payload、session_id、iteration、timestamp 等字段
- 包含 session_id、iteration、type 等元数据
- 支持多种事件类型：started、planning、decision、tool_call、tool_result、done、error
- 事件内容由 Runtime 填充，Observer 决定如何展示
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ai_agent.core.event import Event


class ExecutionStatus(str, Enum):
    """执行状态。"""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RETRY = "retry"


class ExecutionOutcome(str, Enum):
    """执行结果：决定下一步做什么。"""

    CONTINUE = "continue"
    STOP = "stop"
    NEED_HUMAN = "need_human"
    NEED_APPROVAL = "need_approval"
    NEED_SUB_AGENT = "need_sub_agent"


@dataclass(frozen=True)
class ExecutionResult:
    """统一的执行结果封装。

    Runtime 只需要关心：
    - status: 执行是否成功
    - outcome: 是否继续循环
    - output: 执行产出（供 Context 更新用）

    异常由 Executor 包装，Runtime 不需要 try-catch。
    """

    status: ExecutionStatus
    outcome: ExecutionOutcome
    output: Any = None
    error: Optional[str] = None
    action_trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_continue(self) -> bool:
        """是否应该继续循环。"""
        return self.outcome == ExecutionOutcome.CONTINUE

    @property
    def is_success(self) -> bool:
        """是否执行成功。"""
        return self.status == ExecutionStatus.SUCCESS

    @classmethod
    def success(
        cls,
        output: Any,
        outcome: ExecutionOutcome = ExecutionOutcome.CONTINUE,
        **kwargs,
    ) -> "ExecutionResult":
        """创建成功结果。"""
        return cls(
            status=ExecutionStatus.SUCCESS, outcome=outcome, output=output, **kwargs
        )

    @classmethod
    def from_error(
        cls,
        message: str,
        outcome: ExecutionOutcome = ExecutionOutcome.STOP,
        **kwargs,
    ) -> "ExecutionResult":
        """创建错误结果。"""
        return cls(
            status=ExecutionStatus.ERROR, outcome=outcome, error=message, **kwargs
        )

    @classmethod
    def timeout(cls, message: str = "执行超时", **kwargs) -> "ExecutionResult":
        """创建超时结果。"""
        return cls(
            status=ExecutionStatus.TIMEOUT,
            outcome=ExecutionOutcome.STOP,
            error=message,
            **kwargs,
        )

    @classmethod
    def retry(cls, message: str, **kwargs) -> "ExecutionResult":
        """创建重试结果。"""
        return cls(
            status=ExecutionStatus.RETRY,
            outcome=ExecutionOutcome.CONTINUE,
            error=message,
            **kwargs,
        )


class RuntimeEventType(str, Enum):
    """Runtime 事件类型。"""

    STARTED = "agent.started"
    PLANNING = "agent.planning"
    DECISION = "agent.decision"
    TOOL_CALL = "agent.tool_call"
    TOOL_RESULT = "agent.tool_result"
    ITERATION = "agent.iteration"
    DONE = "agent.done"
    ERROR = "agent.error"


@dataclass(frozen=True)
class RuntimeEvent(Event):
    """Runtime 发布的事件。

    继承自 Event，共享 name、payload、session_id、iteration、timestamp 等字段。
    事件内容由 Runtime 填充，Observer 决定如何展示。
    """

    type: Optional[RuntimeEventType] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.type is None:
            raise ValueError("type is required")
        object.__setattr__(self, "name", self.type.value)
        object.__setattr__(self, "payload", self.data)

    @classmethod
    def started(cls, session_id: str, **kwargs) -> "RuntimeEvent":
        """创建启动事件。"""
        return cls(type=RuntimeEventType.STARTED, session_id=session_id, **kwargs)

    @classmethod
    def planning(cls, session_id: str, iteration: int, **kwargs) -> "RuntimeEvent":
        """创建规划事件。"""
        return cls(
            type=RuntimeEventType.PLANNING,
            session_id=session_id,
            iteration=iteration,
            **kwargs,
        )

    @classmethod
    def decision(
        cls, session_id: str, iteration: int, action_type: str, **kwargs
    ) -> "RuntimeEvent":
        """创建决策事件。"""
        return cls(
            type=RuntimeEventType.DECISION,
            session_id=session_id,
            iteration=iteration,
            data={"action_type": action_type, **kwargs},
        )

    @classmethod
    def tool_call(
        cls,
        session_id: str,
        iteration: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        **kwargs,
    ) -> "RuntimeEvent":
        """创建工具调用事件。"""
        return cls(
            type=RuntimeEventType.TOOL_CALL,
            session_id=session_id,
            iteration=iteration,
            data={"tool": tool_name, "args": tool_args, **kwargs},
        )

    @classmethod
    def tool_result(
        cls,
        session_id: str,
        iteration: int,
        tool_name: str,
        observation: Any,
        **kwargs,
    ) -> "RuntimeEvent":
        """创建工具结果事件。"""
        return cls(
            type=RuntimeEventType.TOOL_RESULT,
            session_id=session_id,
            iteration=iteration,
            data={"tool": tool_name, "observation": str(observation), **kwargs},
        )

    @classmethod
    def iteration(cls, session_id: str, iteration: int, **kwargs) -> "RuntimeEvent":
        """创建迭代事件。"""
        return cls(
            type=RuntimeEventType.ITERATION,
            session_id=session_id,
            iteration=iteration,
            **kwargs,
        )

    @classmethod
    def done(
        cls, session_id: str, iteration: int, success: bool, **kwargs
    ) -> "RuntimeEvent":
        """创建完成事件。"""
        return cls(
            type=RuntimeEventType.DONE,
            session_id=session_id,
            iteration=iteration,
            data={"success": success, **kwargs},
        )

    @classmethod
    def create_error(
        cls,
        session_id: str,
        iteration: int,
        message: str,
        **kwargs,
    ) -> "RuntimeEvent":
        """创建错误事件。"""
        return cls(
            type=RuntimeEventType.ERROR,
            session_id=session_id,
            iteration=iteration,
            error=message,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转成字典，方便序列化。"""
        return {
            "type": getattr(self.type, "value", None),
            "session_id": self.session_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "data": self.data,
            "error": self.error,
        }
