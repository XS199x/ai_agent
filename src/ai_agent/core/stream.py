"""StreamHandle：流式输出的"中间层"。

架构：
    生产者（LLM / Agent / Tool）→ StreamHandle.[emit_token|emit_done|emit_event]
                                                   ↓
                                      写入 asyncio.Queue（内部）
                                                   ↓
    消费者（FastAPI / WS）         ← StreamHandle.stream()（异步迭代）

设计原则：
- 一次使用：一个 HTTP 请求一个 StreamHandle；用完不可复用
- 不抛异常：emit 的内部异常只走 EventBus 的 error 事件
- 可序列化：每个 StreamItem 都能转成旧版的 ChatCompletionChunk 字典，保证前端 SSE 格式不变
- 可扩展：emit_event 留给 Agent/Tool 推"正在调搜索/正在思考"等非 token 事件
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.ai_agent.core.event import Event, EventBus

# ---------------------------------------------------------------------------
# StreamItem：流中的一个元素
# ---------------------------------------------------------------------------


@dataclass
class StreamItem:
    kind: str  # "token" | "done" | "error" | "event"
    delta: str = ""
    full_text: str = ""
    event_name: str = ""
    event_payload: Dict[str, Any] = field(default_factory=dict)
    raw_chunk: Optional[Dict[str, Any]] = (
        None  # 兼容旧 SSE：如果是 LLM 原始 chunk，原样保留一份
    )
    created_at: float = field(default_factory=time.time)

    # --- 序列化为 ChatCompletionChunk 格式的 dict，由外部负责 json.dumps ---
    def to_sse_json(self) -> Dict[str, Any]:
        """输出与旧版 SSE 协议一致的 dict 结构。

        外部 item_to_sse_line() 负责把这个 dict 转成 JSON 字符串并包成 data: 行。
        raw_chunk 优先：如果是 LLM SDK 直接给的原始 chunk，就原封不动用它，
        确保协议跟之前完全兼容，前端不需要改。
        """
        if self.raw_chunk is not None:
            return dict(self.raw_chunk)

        # 模拟 ChatCompletionChunk 结构
        if self.kind == "token":
            return {
                "id": "steam-chunk",
                "object": "chat.completion.chunk",
                "created": int(self.created_at),
                "model": "",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": self.delta},
                        "finish_reason": None,
                    }
                ],
            }
        if self.kind == "done":
            return {
                "id": "steam-chunk",
                "object": "chat.completion.chunk",
                "created": int(self.created_at),
                "model": "",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
        if self.kind == "error":
            return {
                "id": "steam-chunk",
                "object": "chat.completion.chunk",
                "created": int(self.created_at),
                "model": "",
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
            }
        # event：以"_event"字段标记，让未来前端做分支
        return {
            "id": "steam-chunk",
            "object": "chat.completion.chunk",
            "created": int(self.created_at),
            "model": "",
            "_event": {"name": self.event_name, "payload": self.event_payload},
            "choices": [],
        }


# ---------------------------------------------------------------------------
# StreamHandle：一次流式会话
# ---------------------------------------------------------------------------


class StreamHandle:
    _SENTINEL_DONE = object()

    def __init__(
        self, bus: Optional[EventBus] = None, session_id: Optional[str] = None
    ) -> None:
        self._bus = bus
        self._session_id = session_id
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._done = False
        self._full_text: str = ""
        self._token_count: int = 0
        self._has_produced: bool = False

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------
    @property
    def full_text(self) -> str:
        return self._full_text

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # 生产者接口
    # ------------------------------------------------------------------
    def emit_token(
        self, delta: str, raw_chunk: Optional[Dict[str, Any]] = None
    ) -> None:
        """向流中写入一个 token。delta 为空会被忽略。"""
        if self._done:
            return
        if not delta:
            return
        self._full_text += delta
        self._token_count += 1
        self._has_produced = True
        item = StreamItem(kind="token", delta=delta, raw_chunk=raw_chunk)
        self._queue.put_nowait(item)
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
        item = StreamItem(
            kind="done",
            full_text=self._full_text,
            event_payload={"token_count": self._token_count, **(meta or {})},
        )
        self._queue.put_nowait(item)
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

    def emit_error(self, message: str) -> None:
        if self._done:
            return
        self._done = True
        item = StreamItem(kind="error", event_payload={"message": message})
        self._queue.put_nowait(item)
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
        merged_payload = {"session_id": self._session_id, **(payload or {})}
        item = StreamItem(kind="event", event_name=name, event_payload=merged_payload)
        self._queue.put_nowait(item)
        if self._bus is not None:
            self._bus.emit(Event(name=name, payload=merged_payload))

    # ------------------------------------------------------------------
    # 便捷方法：消费 LLM 的 chunk 流
    # ------------------------------------------------------------------
    async def consume_llm_chunk_stream(
        self,
        chunk_stream: AsyncGenerator[Any, None],
        pass_raw_chunk: bool = True,
    ) -> None:
        """把 LLM 的 AsyncGenerator[ChatCompletionChunk] 流转成 StreamHandle 的 token 流。

        - 迭代 chunk 流
        - 从每个 chunk.delta.content 提取 delta，调用 self.emit_token()
        - 正常结束或异常都会自动 emit_done / emit_error

        参数：
            pass_raw_chunk：True 时保留 chunk.model_dump()，以便序列化时
                             跟旧流程输出"尽量一致"（默认 True，对向后兼容最友好）。
        """
        try:
            if self._bus is not None:
                self._bus.emit(
                    Event(name="llm.start", payload={"session_id": self._session_id})
                )

            async for chunk in chunk_stream:
                delta = self._extract_delta(chunk)
                if pass_raw_chunk:
                    raw = getattr(chunk, "model_dump", None)
                    raw_dict = raw() if raw is not None else None
                else:
                    raw_dict = None
                self.emit_token(delta, raw_chunk=raw_dict)
        except Exception as e:
            self.emit_error(f"{type(e).__name__}: {e}")
            return

        self.emit_done()

    @staticmethod
    def _extract_delta(chunk: Any) -> str:
        """从各种 chunk 对象里取 content 文本，写得鲁棒点。"""
        try:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                return ""
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                return ""
            content = getattr(delta, "content", None)
            return content or ""
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 消费者接口
    # ------------------------------------------------------------------
    async def stream(self) -> AsyncGenerator[StreamItem, None]:
        """返回一个异步迭代器，直到遇到"结束标记"才停止。"""
        while True:
            item = await self._queue.get()
            if item is self._SENTINEL_DONE:
                return
            yield item


# ---------------------------------------------------------------------------
# 便捷序列化：把 StreamItem 转成前端 SSE 用的 JSON 字符串
# ---------------------------------------------------------------------------


def item_to_sse_line(item: StreamItem) -> str:
    """输出形如 'data: {...}\n\n' 的字符串，可直接 yield 给 StreamingResponse。"""
    body = json.dumps(item.to_sse_json(), ensure_ascii=False)
    return f"data: {body}\n\n"
