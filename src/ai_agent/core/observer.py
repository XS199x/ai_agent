import asyncio

from ai_agent.core.event import EventLike
from ai_agent.core.stream import StreamHandle
from ai_agent.models.runtime import RuntimeEventType

_TOKEN_DELAY_PER_CHAR = 0.020


class StreamEventObserver:
    def __init__(self, handle: StreamHandle) -> None:
        self._handle = handle
        self._replay_task: asyncio.Task | None = None

    def __call__(self, event: EventLike) -> None:
        if self._handle.done:
            return

        name = event.name
        payload = dict(event.payload)
        if event.error is not None:
            payload["error"] = event.error

        if name == RuntimeEventType.DONE.value:
            if payload.get("success", False):
                answer = payload.get("answer") or payload.get("message") or ""
                if answer and not self._handle.has_produced:
                    self._replay_task = asyncio.create_task(
                        self._replay_with_delay(answer, payload)
                    )
                else:
                    self._handle.finish_success(**payload)
            else:
                self._handle.finish_error(payload.get("message", "unknown error"))

        elif name == RuntimeEventType.ERROR.value:
            self._handle.emit_error(payload.get("message", "unknown error"))

        elif name == RuntimeEventType.TOKEN.value:
            self._handle.emit_token(payload.get("delta", ""))

        elif name == RuntimeEventType.LLM_DONE.value:
            self._handle.emit_done(payload)

        elif name and name.startswith("agent."):
            self._handle.emit_event(name, payload)

    async def _replay_with_delay(self, text: str, finish_kwargs: dict) -> None:
        try:
            for ch in text:
                if self._handle.done:
                    return
                self._handle.emit_token(ch)
                await asyncio.sleep(_TOKEN_DELAY_PER_CHAR)
        finally:
            if not self._handle.done:
                self._handle.finish_success(**finish_kwargs)
