import asyncio
import traceback
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from ai_agent.models.runtime import Event


Handler = Callable[["Event"], object]


class EventBus:
    def __init__(self, handlers: Optional[List[Handler]] = None) -> None:
        self._handlers: List[Handler] = list(handlers) if handlers else []

    def subscribe(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def emit(self, event: "Event") -> None:
        for h in list(self._handlers):
            try:
                result = h(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                traceback.print_exc()

    async def emit_async(self, event: "Event") -> None:
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


__all__ = ["EventBus", "get_default_bus", "Handler"]
