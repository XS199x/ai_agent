"""对话持久化存储：基于 SQLite 的线程安全实现。"""

import json
import os
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from ai_agent.models.chat import ChatMessage, FunctionCall, ToolCall

from .models import Conversation

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    session_id    TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    system_prompt TEXT,
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT,
    extra_json   TEXT,
    created_at   INTEGER NOT NULL,
    FOREIGN KEY(session_id) REFERENCES conversations(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id ASC);
"""


class ConversationStore:
    """线程安全的 SQLite 会话存储。"""

    def __init__(
        self,
        max_conversations: int = 100,
        persist_path: Optional[os.PathLike | str] = None,
    ) -> None:
        self._max = max_conversations
        self._lock = RLock()
        self._persist_path: Optional[Path] = (
            Path(persist_path).expanduser().resolve() if persist_path else None
        )

        self._store: Dict[str, Conversation] = {}
        self._order: List[str] = []

        if self._persist_path is not None:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._persist_path),
                check_same_thread=False,
                isolation_level=None,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            with self._lock:
                self._conn.executescript(SCHEMA)
                self._load_all()
        else:
            self._conn = None

    def _load_all(self) -> None:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT session_id, title, system_prompt, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
        for r in rows:
            conv = Conversation(
                session_id=r["session_id"],
                title=r["title"],
                system_prompt=r["system_prompt"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            mcur = self._conn.execute(
                "SELECT role, content, extra_json FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (r["session_id"],),
            )
            for mr in mcur.fetchall():
                extra_data = {}
                if mr["extra_json"]:
                    try:
                        extra_data = json.loads(mr["extra_json"])
                    except Exception:
                        pass

                tool_calls_data = extra_data.get("tool_calls", [])
                tool_calls = []
                for tc_data in tool_calls_data:
                    try:
                        func_data = tc_data.get("function", {})
                        tool_calls.append(
                            ToolCall(
                                id=tc_data.get("id", ""),
                                type=tc_data.get("type", "function"),
                                function=FunctionCall(
                                    name=func_data.get("name", ""),
                                    arguments=func_data.get("arguments", "{}"),
                                ),
                            )
                        )
                    except Exception:
                        pass

                msg = ChatMessage(
                    role=mr["role"],
                    content=mr["content"],
                    name=extra_data.get("name"),
                    tool_call_id=extra_data.get("tool_call_id"),
                    tool_calls=tool_calls if tool_calls else None,
                )
                conv.messages.append(msg)
            self._store[conv.session_id] = conv
            self._order.append(conv.session_id)

    def _touch_updated_at(self, session_id: str) -> None:
        if self._conn is None:
            return
        now = int(time.time())
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        conv = self._store.get(session_id)
        if conv is not None:
            conv.updated_at = now

    def _trim(self) -> None:
        while len(self._order) > self._max:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)
            if self._conn is not None:
                self._conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (oldest,),
                )

    def create(
        self,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Conversation:
        with self._lock:
            conv = Conversation(
                title=title or "新对话",
                system_prompt=system_prompt,
            )
            if self._conn is not None:
                self._conn.execute(
                    "INSERT INTO conversations(session_id, title, system_prompt, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        conv.session_id,
                        conv.title,
                        conv.system_prompt,
                        conv.created_at,
                        conv.updated_at,
                    ),
                )
            self._store[conv.session_id] = conv
            self._order.append(conv.session_id)
            self._trim()
            return conv

    def rename(self, session_id: str, title: str) -> Optional[Conversation]:
        with self._lock:
            conv = self._store.get(session_id)
            if conv is None:
                return None
            conv.rename(title)
            if self._conn is not None:
                self._conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title, conv.updated_at, session_id),
                )
            return conv

    def update_system_prompt(
        self, session_id: str, system_prompt: Optional[str]
    ) -> Optional[Conversation]:
        with self._lock:
            conv = self._store.get(session_id)
            if conv is None:
                return None
            conv.system_prompt = system_prompt
            conv.updated_at = int(time.time())
            if self._conn is not None:
                self._conn.execute(
                    "UPDATE conversations SET system_prompt = ?, updated_at = ? WHERE session_id = ?",
                    (system_prompt, conv.updated_at, session_id),
                )
            return conv

    def clear_messages(self, session_id: str) -> Optional[Conversation]:
        with self._lock:
            conv = self._store.get(session_id)
            if conv is None:
                return None
            conv.clear()
            if self._conn is not None:
                self._conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session_id,),
                )
                self._touch_updated_at(session_id)
            return conv

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._store:
                return False
            del self._store[session_id]
            try:
                self._order.remove(session_id)
            except ValueError:
                pass
            if self._conn is not None:
                self._conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,),
                )
            return True

    def append_message(
        self, session_id: str, message: ChatMessage
    ) -> Optional[Conversation]:
        with self._lock:
            conv = self._store.get(session_id)
            if conv is None:
                return None
            conv.append(message)
            if self._conn is not None:
                extra_data = {}
                if message.name:
                    extra_data["name"] = message.name
                if message.tool_call_id:
                    extra_data["tool_call_id"] = message.tool_call_id
                if message.tool_calls:
                    extra_data["tool_calls"] = [
                        tc.model_dump() for tc in message.tool_calls
                    ]
                extra_json = json.dumps(extra_data) if extra_data else None
                self._conn.execute(
                    "INSERT INTO messages(session_id, role, content, extra_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        message.role,
                        message.content or "",
                        extra_json,
                        conv.updated_at,
                    ),
                )
                self._touch_updated_at(session_id)
            return conv

    def extend_messages(
        self, session_id: str, messages: List[ChatMessage]
    ) -> Optional[Conversation]:
        with self._lock:
            conv = self._store.get(session_id)
            if conv is None:
                return None
            conv.extend(messages)
            if self._conn is not None:
                now = int(time.time())
                rows = []
                for m in messages:
                    extra_data = {}
                    if m.name:
                        extra_data["name"] = m.name
                    if m.tool_call_id:
                        extra_data["tool_call_id"] = m.tool_call_id
                    if m.tool_calls:
                        extra_data["tool_calls"] = [
                            tc.model_dump() for tc in m.tool_calls
                        ]
                    extra_json = json.dumps(extra_data) if extra_data else None
                    rows.append((session_id, m.role, m.content or "", extra_json, now))
                self._conn.executemany(
                    "INSERT INTO messages(session_id, role, content, extra_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                self._touch_updated_at(session_id)
            return conv

    def get(self, session_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._store.get(session_id)

    def list_all(self) -> List[Conversation]:
        with self._lock:
            items = [self._store[sid] for sid in self._order if sid in self._store]
            items.sort(key=lambda c: c.updated_at, reverse=True)
            return items


__all__ = ["ConversationStore"]
