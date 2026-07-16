import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional

from ai_agent.core.event import Event, EventBus


@dataclass
class StreamItem:
    kind: str  # "token" | "done" | "error" | "event"
    delta: str = ""
    full_text: str = ""
    event_name: str = ""
    event_payload: Dict[str, Any] = field(default_factory=dict)
    raw_chunk: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)

    def to_sse_json(self) -> Dict[str, Any]:
        if self.raw_chunk is not None:
            return dict(self.raw_chunk)

        base: Dict[str, Any] = {
            "id": "stream-chunk",
            "object": "chat.completion.chunk",
            "created": int(self.created_at),
            "model": "",
        }

        kind_map = {
            "token": lambda: {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": self.delta},
                        "finish_reason": None,
                    }
                ]
            },
            "done": lambda: {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            },
            "error": lambda: {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": self.event_payload.get("message", "error"),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "_error": True,
            },
            "event": lambda: {
                "choices": [],
                "_event": {"name": self.event_name, "payload": self.event_payload},
            },
        }

        base.update(kind_map.get(self.kind, kind_map["event"])())
        return base


class StreamHandle:
    _SENTINEL_DONE = object()

    def __init__(
        self, bus: Optional[EventBus] = None, session_id: Optional[str] = None
    ) -> None:
        self._bus = bus
        self._session_id = session_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = False
        self._full_text: str = ""
        self._token_count: int = 0
        self._start_time: float = 0.0
        self._success: bool = False
        self._error: Optional[str] = None
        self.has_produced: bool = False

    @property
    def full_text(self) -> str:
        return self._full_text

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def done(self) -> bool:
        return self._done

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def success(self) -> bool:
        return self._success

    @property
    def error(self) -> Optional[str]:
        return self._error

    def emit_token(
        self, delta: str, raw_chunk: Optional[Dict[str, Any]] = None
    ) -> None:
        if self._done or not delta:
            return
        self._full_text += delta
        self._token_count += 1
        self.has_produced = True
        self._queue.put_nowait(
            StreamItem(kind="token", delta=delta, raw_chunk=raw_chunk)
        )
        if self._bus is not None:
            self._bus.emit(
                Event(
                    name="llm.token",
                    payload={"session_id": self._session_id, "delta_len": len(delta)},
                )
            )

    def emit_done(self, meta: Optional[Dict[str, Any]] = None) -> None:
        if self._done:
            return
        self._done = True
        self._queue.put_nowait(
            StreamItem(
                kind="done",
                full_text=self._full_text,
                event_payload={"token_count": self._token_count, **(meta or {})},
            )
        )
        if self._bus is not None:
            self._bus.emit(
                Event(
                    name="llm.done",
                    payload={
                        "session_id": self._session_id,
                        "token_count": self._token_count,
                        "len": len(self._full_text),
                        **(meta or {}),
                    },
                )
            )
        self._queue.put_nowait(self._SENTINEL_DONE)

    def finish_success(self, **kwargs) -> None:
        self._success = True
        self.emit_done({"success": True, **kwargs})

    def finish_error(self, message: str) -> None:
        self._error = message
        self.emit_error(message)

    def emit_error(self, message: str) -> None:
        if self._done:
            return
        self._done = True
        self._queue.put_nowait(
            StreamItem(kind="error", event_payload={"message": message})
        )
        if self._bus is not None:
            self._bus.emit(
                Event(
                    name="llm.error",
                    payload={"session_id": self._session_id, "message": message},
                )
            )
        self._queue.put_nowait(self._SENTINEL_DONE)

    def emit_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self._done:
            return
        merged = {"session_id": self._session_id, **(payload or {})}
        self._queue.put_nowait(
            StreamItem(kind="event", event_name=name, event_payload=merged)
        )
        if self._bus is not None:
            self._bus.emit(Event(name=name, payload=merged))

    async def stream(self) -> AsyncGenerator[StreamItem, None]:
        while True:
            item = await self._queue.get()
            if item is self._SENTINEL_DONE:
                return
            yield item


def item_to_sse_line(item: StreamItem) -> str:
    body = json.dumps(item.to_sse_json(), ensure_ascii=False)
    return f"data: {body}\n\n"
