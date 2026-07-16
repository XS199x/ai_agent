"""事件处理器：订阅 EventBus 并处理特定事件。"""

import asyncio
import json
import os
import traceback
from datetime import datetime
from typing import Any, Optional

from ai_agent.core.event import Event


class PrintLogHandler:
    """带时间戳的控制台日志。"""

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
        print(f"[{ts}] {icon} {event.name:<28}{payload_text}")


class StructuredLogHandler:
    """结构化 JSON 日志。"""

    _LEVEL_MAP = {
        "agent.error": "error",
        "agent.tool_error": "error",
        "llm.done": "info",
        "chat.prompt_stats": "info",
        "agent.planning.decision": "info",
    }

    def __init__(self, skip_tokens: bool = True, indent: bool = False) -> None:
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
    """把事件写入 JSONL 文件。"""

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


class ConversationPersistHandler:
    """订阅 conversation.append 事件，将消息写入 ConversationStore。"""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __call__(self, event: Event) -> None:
        if event.name != "conversation.append":
            return

        session_id = event.payload.get("session_id")
        message = event.payload.get("message")

        if session_id and message:
            try:
                self._store.append_message(session_id, message)
            except Exception:
                traceback.print_exc()


class TokenCountHandler:
    """追踪每次请求的 prompt token 与最终回复 token。"""

    def __init__(self) -> None:
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
            existing = self._latest.setdefault(sid, {})
            existing["completion_tokens"] = event.payload.get("token_count", 0)
            existing["ts"] = event.timestamp

    def get_latest(self, session_id: Optional[str] = None) -> Optional[dict]:
        if session_id is None:
            return dict(self._latest)
        return self._latest.get(session_id)


__all__ = [
    "PrintLogHandler",
    "StructuredLogHandler",
    "FileLogHandler",
    "ConversationPersistHandler",
    "TokenCountHandler",
]
