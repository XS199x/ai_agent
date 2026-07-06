"""EventBus：把"流程中发生的事"发布给关心它的人。

设计原则：
- 极简：只有 Event / EventBus 两个类，subscribe / emit 两个主要方法
- 容错：单个 handler 抛异常不中断主流程（打印到 stderr）
- 同步优先：emit 同步调用所有 handler；若需异步，用 emit_async（await 每个 async handler）

新特性（A3）：
- 新增 FileLogHandler：把事件写入文件，方便回溯
- 新增 TokenCountHandler：统计 prompt / llm 完成时累计 token

典型用法：

    bus = EventBus()
    bus.subscribe(PrintLogHandler())
    bus.subscribe(FileLogHandler(path="logs/events.log"))
    bus.subscribe(TokenCountHandler())

    bus.emit(Event("llm.start", {"model": "deepseek-chat"}))
"""

import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Any, Awaitable, Callable, List, Optional


@dataclass
class Event:
    name: str
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "error": self.error,
        }


Handler = Callable[[Event], Any]
AsyncHandler = Callable[[Event], Awaitable[Any]]


class EventBus:
    def __init__(self, handlers: Optional[List[Handler]] = None) -> None:
        self._handlers: List[Handler] = list(handlers) if handlers else []

    # --- 订阅 ---
    def subscribe(self, handler: Handler) -> None:
        """注册一个 handler。handler 可以是同步函数、也可以是 async def（交给 emit_async 时才 await）。"""
        self._handlers.append(handler)

    def unsubscribe(self, handler: Handler) -> None:
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    # --- 发布（同步：所有同步 handler 立即调用；async handler 不 await，避免阻塞）---
    def emit(self, event: Event) -> None:
        import asyncio

        for h in list(self._handlers):
            try:
                result = h(event)
                if asyncio.iscoroutine(result):
                    # 同步发布场景下，async handler 不阻塞调用者
                    # 这里"忘掉"它（让它在某个事件循环里跑，由调用方负责）
                    # 若需要严格 await，请使用 emit_async
                    pass
            except Exception:
                traceback.print_exc()

    # --- 发布（异步：await 所有 async handler）---
    async def emit_async(self, event: Event) -> None:
        import asyncio

        for h in list(self._handlers):
            try:
                result = h(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                traceback.print_exc()

    def __contains__(self, handler: Handler) -> bool:
        return handler in self._handlers

    def __len__(self) -> int:
        return len(self._handlers)


# ---------------------------------------------------------------------------
# 内置 handlers
# ---------------------------------------------------------------------------


class PrintLogHandler:
    """Phase 2 升级版：带时间戳、分等级、更紧凑美观。

    输出格式：
        [HH:MM:SS] ⚡ agent.planning          iteration=1 available_tools=3
        [HH:MM:SS] ✓ agent.tool_call          tool=calculator
        [HH:MM:SS] ✓ agent.tool_result        tool=calculator duration_ms=0
        [HH:MM:SS] → chat.prompt_stats        prompt_tokens=42

    工具调用用 ✓ / ✗ 标识成功/失败；纯信息类用 →；Planner 用 ⚡。
    """

    # 事件名 → ASCII 图标（避免 Windows gbk 终端编码问题）
    _ICON_MAP = {
        "agent.planning": "* ",
        "agent.planning.decision": "= ",
        "agent.decision": "> ",
        "agent.tool_call": "-> ",
        "agent.tool_result": "OK ",
        "agent.tool_error": "!! ",
        "agent.error": "!! ",
        "chat.prompt_stats": "-> ",
        "llm.start": ". ",
        "llm.done": "OK ",
    }

    def __init__(self, skip_tokens: bool = True) -> None:
        self.skip_tokens = skip_tokens

    @staticmethod
    def _format_payload(payload: dict) -> str:
        """把 payload dict 变成 key=value 紧凑形式。"""
        if not payload:
            return ""
        parts = []
        for k, v in payload.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v}")
            elif isinstance(v, list):
                parts.append(f"{k}=[{', '.join(str(x) for x in v[:10])}]")
            else:
                s = str(v)
                if len(s) > 40:
                    s = s[:37] + "..."
                parts.append(f"{k}={s}")
        return "  ".join(parts)

    def __call__(self, event: Event) -> None:
        if self.skip_tokens and event.name == "llm.token":
            return
        ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        icon = self._ICON_MAP.get(event.name, "• ")
        payload_text = self._format_payload(event.payload)
        # 左对齐事件名，固定宽度 28，方便对齐
        print(f"[{ts}] {icon} {event.name:<28}{payload_text}")


class StructuredLogHandler:
    """Phase 2 新增：结构化 JSON 日志，一行一个事件，方便 grep/分析。

    输出示例（一行）：
        {"ts": "2026-07-02T17:30:00", "level": "info", "name": "agent.tool_result",
         "payload": {"tool": "calculator", "success": true, "duration_ms": 0}}
    """

    # 事件名 → 日志级别
    _LEVEL_MAP = {
        "agent.error": "error",
        "agent.tool_error": "error",
        "llm.done": "info",
        "chat.prompt_stats": "info",
        "agent.planning.decision": "info",
    }

    def __init__(
        self,
        skip_tokens: bool = True,
        indent: bool = False,
    ) -> None:
        self.skip_tokens = skip_tokens
        self.indent = indent

    def __call__(self, event: Event) -> None:
        if self.skip_tokens and event.name == "llm.token":
            return
        line = {
            "ts": datetime.fromtimestamp(event.timestamp).isoformat(timespec="seconds"),
            "level": self._LEVEL_MAP.get(event.name, "info"),
            "name": event.name,
            "payload": event.payload,
        }
        if self.indent:
            print(json.dumps(line, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(line, ensure_ascii=False))


class FileLogHandler:
    """把事件写入 JSONL 文件，方便事后回放。"""

    def __init__(
        self,
        path: str = "logs/events.log",
        skip_tokens: bool = True,
        max_lines: int = 10000,
    ) -> None:
        self.path = path
        self.skip_tokens = skip_tokens
        self.max_lines = max_lines
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        if self.skip_tokens and event.name == "llm.token":
            return
        line = {
            "ts": datetime.fromtimestamp(event.timestamp).isoformat(timespec="seconds"),
            "name": event.name,
            "payload": event.payload,
        }
        try:
            self._fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            if not getattr(self, "_fh", None) or self._fh.closed:
                return
            self._fh.close()
        except Exception:
            pass


class TokenCountHandler:
    """追踪每次请求的 prompt token 与最终回复 token，用于成本估算 / 前端展示。

    - chat.prompt_stats 事件：记录本轮 prompt token
    - llm.done 事件：记录本轮完成时的总 token 估计
    """

    def __init__(self) -> None:
        # 按 session_id 存储最近一次统计；{ session_id: { prompt_tokens, completion_tokens, ts } }
        self._latest: dict = {}

    def __call__(self, event: Event) -> None:
        sid = event.payload.get("session_id") or "_global_"
        if event.name == "chat.prompt_stats":
            self._latest[sid] = {
                "prompt_tokens": event.payload.get("prompt_tokens", 0),
                "completion_tokens": event.payload.get("completion_tokens", 0),
                "ts": event.timestamp,
            }
        elif event.name == "llm.done":
            # llm.done 事件里 StreamHandle 已附带 token_count
            existing = self._latest.setdefault(sid, {})
            existing["completion_tokens"] = event.payload.get("token_count", 0)
            existing["ts"] = event.timestamp

    def get_latest(self, session_id: Optional[str] = None) -> Optional[dict]:
        """返回最近一次统计（如果有的话）。不指定 session_id 时返回一个按 session_id 聚合的总览。"""
        if session_id is None:
            return dict(self._latest)
        return self._latest.get(session_id)


# 全局默认 bus，app.py 会用它
_default_bus: Optional[EventBus] = None


def get_default_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
        _default_bus.subscribe(PrintLogHandler())
        _default_bus.subscribe(TokenCountHandler())
    return _default_bus
