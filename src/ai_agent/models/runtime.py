"""Runtime core models: define execution results and runtime events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, Optional


class ExecutionStatus(str, Enum):
    """Execution status."""

    SUCCESS = "success"
    ERROR = "error"


class ExecutionOutcome(str, Enum):
    """Execution outcome: decides what to do next."""

    CONTINUE = "continue"
    STOP = "stop"


@dataclass(frozen=True)
class ExecutionResult:
    """Unified execution result wrapper."""

    status: ExecutionStatus
    outcome: ExecutionOutcome
    output: Any = None
    error: Optional[str] = None
    action_trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_continue(self) -> bool:
        return self.outcome == ExecutionOutcome.CONTINUE

    @property
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    @classmethod
    def success(
        cls,
        output: Any,
        outcome: ExecutionOutcome = ExecutionOutcome.CONTINUE,
        **kwargs,
    ) -> "ExecutionResult":
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
        return cls(
            status=ExecutionStatus.ERROR, outcome=outcome, error=message, **kwargs
        )


class RuntimeEventType(str, Enum):
    """Runtime event types."""

    STARTED = "agent.started"
    DECISION = "agent.decision"
    TOOL_CALL = "agent.tool_call"
    TOOL_RESULT = "agent.tool_result"
    ITERATION = "agent.iteration"
    DONE = "agent.done"
    ERROR = "agent.error"
    TOKEN = "llm.token"
    LLM_DONE = "llm.done"
    LLM_ERROR = "llm.error"


@dataclass(frozen=True)
class RuntimeEvent:
    """Runtime event with strong type safety.

    Independent from Event, has its own field definitions matching
    Event's required interface for EventBus compatibility.
    """

    type: RuntimeEventType
    session_id: Optional[str] = None
    iteration: int = 0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time)
    _data: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.type.value

    @property
    def payload(self) -> Dict[str, Any]:
        data = dict(self._data)
        if self.error is not None:
            data["error"] = self.error
        return data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "data": self._data,
            "error": self.error,
        }

    @classmethod
    def started(cls, session_id: str, **kwargs) -> "RuntimeEvent":
        return cls(type=RuntimeEventType.STARTED, session_id=session_id, **kwargs)

    @classmethod
    def decision(
        cls, session_id: str, iteration: int, action_type: str, **kwargs
    ) -> "RuntimeEvent":
        return cls(
            type=RuntimeEventType.DECISION,
            session_id=session_id,
            iteration=iteration,
            _data={"action_type": action_type, **kwargs},
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
        return cls(
            type=RuntimeEventType.TOOL_CALL,
            session_id=session_id,
            iteration=iteration,
            _data={"tool": tool_name, "args": tool_args, **kwargs},
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
        return cls(
            type=RuntimeEventType.TOOL_RESULT,
            session_id=session_id,
            iteration=iteration,
            _data={"tool": tool_name, "observation": str(observation), **kwargs},
        )

    @classmethod
    def iteration_event(
        cls, session_id: str, iteration: int, **kwargs
    ) -> "RuntimeEvent":
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
        return cls(
            type=RuntimeEventType.DONE,
            session_id=session_id,
            iteration=iteration,
            _data={"success": success, **kwargs},
        )

    @classmethod
    def create_error(
        cls, session_id: str, iteration: int, message: str, **kwargs
    ) -> "RuntimeEvent":
        return cls(
            type=RuntimeEventType.ERROR,
            session_id=session_id,
            iteration=iteration,
            error=message,
            **kwargs,
        )

    @classmethod
    def llm_token(
        cls, session_id: Optional[str], delta_len: int, **kwargs
    ) -> "RuntimeEvent":
        return cls(
            type=RuntimeEventType.TOKEN,
            session_id=session_id,
            _data={"delta_len": delta_len, **kwargs},
        )

    @classmethod
    def llm_done(
        cls, session_id: Optional[str], token_count: int, full_text_len: int, **kwargs
    ) -> "RuntimeEvent":
        return cls(
            type=RuntimeEventType.LLM_DONE,
            session_id=session_id,
            _data={"token_count": token_count, "len": full_text_len, **kwargs},
        )

    @classmethod
    def llm_error(
        cls, session_id: Optional[str], message: str, **kwargs
    ) -> "RuntimeEvent":
        return cls(
            type=RuntimeEventType.LLM_ERROR,
            session_id=session_id,
            error=message,
            **kwargs,
        )
