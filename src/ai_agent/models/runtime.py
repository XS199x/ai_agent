"""Runtime core models: define execution results and the unified event type."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, Optional


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class ExecutionOutcome(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"


@dataclass(frozen=True)
class ExecutionResult:
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


class EventName:
    """Event name constants for type-safe event matching."""

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
class Event:
    """Unified event type for EventBus.

    All events have a name and optional payload. The name is always a string,
    either from EventName constants or a custom identifier.
    """

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    iteration: int = 0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time)

    @classmethod
    def started(cls, session_id: str, **kwargs) -> "Event":
        return cls(name=EventName.STARTED, session_id=session_id, payload=kwargs)

    @classmethod
    def decision(
        cls, session_id: str, iteration: int, action_type: str, **kwargs
    ) -> "Event":
        return cls(
            name=EventName.DECISION,
            session_id=session_id,
            iteration=iteration,
            payload={"action_type": action_type, **kwargs},
        )

    @classmethod
    def tool_call(
        cls,
        session_id: str,
        iteration: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        **kwargs,
    ) -> "Event":
        return cls(
            name=EventName.TOOL_CALL,
            session_id=session_id,
            iteration=iteration,
            payload={"tool": tool_name, "args": tool_args, **kwargs},
        )

    @classmethod
    def tool_result(
        cls,
        session_id: str,
        iteration: int,
        tool_name: str,
        observation: Any,
        **kwargs,
    ) -> "Event":
        return cls(
            name=EventName.TOOL_RESULT,
            session_id=session_id,
            iteration=iteration,
            payload={"tool": tool_name, "observation": str(observation), **kwargs},
        )

    @classmethod
    def iteration_event(cls, session_id: str, iteration: int, **kwargs) -> "Event":
        return cls(
            name=EventName.ITERATION,
            session_id=session_id,
            iteration=iteration,
            payload=kwargs,
        )

    @classmethod
    def done(cls, session_id: str, iteration: int, success: bool, **kwargs) -> "Event":
        return cls(
            name=EventName.DONE,
            session_id=session_id,
            iteration=iteration,
            payload={"success": success, **kwargs},
        )

    @classmethod
    def create_error(
        cls, session_id: str, iteration: int, message: str, **kwargs
    ) -> "Event":
        return cls(
            name=EventName.ERROR,
            session_id=session_id,
            iteration=iteration,
            error=message,
            payload=kwargs,
        )

    @classmethod
    def llm_token(cls, session_id: Optional[str], delta_len: int, **kwargs) -> "Event":
        return cls(
            name=EventName.TOKEN,
            session_id=session_id,
            payload={"delta_len": delta_len, **kwargs},
        )

    @classmethod
    def llm_done(
        cls, session_id: Optional[str], token_count: int, full_text_len: int, **kwargs
    ) -> "Event":
        return cls(
            name=EventName.LLM_DONE,
            session_id=session_id,
            payload={"token_count": token_count, "len": full_text_len, **kwargs},
        )

    @classmethod
    def llm_error(cls, session_id: Optional[str], message: str, **kwargs) -> "Event":
        return cls(
            name=EventName.LLM_ERROR,
            session_id=session_id,
            error=message,
            payload=kwargs,
        )
