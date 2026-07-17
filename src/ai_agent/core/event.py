import asyncio
import traceback
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class EventLike(Protocol):
    name: str
    payload: dict
    error: Optional[str] = None


@dataclass(frozen=True)
class Event:
    name: Optional[str] = None
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time)
    error: Optional[str] = None
    session_id: Optional[str] = None
    iteration: int = 0


Handler = Callable[[EventLike], Any]


class EventBus:
    def __init__(self, handlers: Optional[List[Handler]] = None) -> None:
        self._handlers: List[Handler] = list(handlers) if handlers else []

    def subscribe(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def emit(self, event: EventLike) -> None:
        for h in list(self._handlers):
            try:
                result = h(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                traceback.print_exc()

    async def emit_async(self, event: EventLike) -> None:
        for h in list(self._handlers):
            try:
                result = h(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                traceback.print_exc()


_default_bus: Optional[EventBus] = None


def get_default_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
        from ai_agent.core.handlers import PrintLogHandler, TokenCountHandler

        _default_bus.subscribe(PrintLogHandler())
        _default_bus.subscribe(TokenCountHandler())
    return _default_bus


__all__ = ["Event", "EventLike", "EventBus", "get_default_bus"]
