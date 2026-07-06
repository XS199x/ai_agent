import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

from ai_agent.models.chat import ChatMessage


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
    content      TEXT NOT NULL,
    extra_json   TEXT,
    created_at   INTEGER NOT NULL,
    FOREIGN KEY(session_id) REFERENCES conversations(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id ASC);
"""


class Conversation:
    """表示一个独立的会话，包含历史消息与元数据。"""

    def __init__(
        self,
        session_id: Optional[str] = None,
        title: str = "新对话",
        system_prompt: Optional[str] = None,
        created_at: Optional[int] = None,
        updated_at: Optional[int] = None,
    ) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex
        self.title: str = title
        self.system_prompt: Optional[str] = system_prompt
        self.messages: List[ChatMessage] = []
        now = int(time.time())
        self.created_at: int = created_at if created_at else now
        self.updated_at: int = updated_at if updated_at else now

    def rename(self, title: str) -> None:
        self.title = title
        self.updated_at = int(time.time())

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self.updated_at = int(time.time())

    def extend(self, messages: List[ChatMessage]) -> None:
        self.messages.extend(messages)
        self.updated_at = int(time.time())

    def clear(self) -> None:
        self.messages = []
        self.updated_at = int(time.time())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "system_prompt": self.system_prompt,
            "messages": [m.model_dump() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ConversationStore:
    """线程安全的 SQLite 会话存储。

    - 创建/获取/列表/重命名/更新 system_prompt/清空消息/删除/追加消息 都会立即落盘；
    - 若存在旧的 JSON 文件（`{db_path}.json` 或同目录下 `conversations.json`），
      初始化时会自动迁移一次并改名备份；
    - persist_path 为 None 时退化为纯内存（主要用于测试/旧兼容）。
    """

    def __init__(
        self,
        max_conversations: int = 100,
        persist_path: Optional[os.PathLike | str] = None,
    ) -> None:
        self._max = max_conversations
        self._lock = RLock()
        self._persist_path: Optional[Path] = (
            Path(persist_path).expanduser().resolve()
            if persist_path
            else None
        )

        # 内存索引：保证读取路径统一
        self._store: Dict[str, Conversation] = {}
        self._order: List[str] = []

        if self._persist_path is not None:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._persist_path),
                check_same_thread=False,
                isolation_level=None,  # 我们用显式事务
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            with self._lock:
                self._conn.executescript(SCHEMA)
                self._maybe_migrate_json()
                self._load_all()
        else:
            self._conn = None

    # ------------------------------------------------------------------
    # SQLite 辅助
    # ------------------------------------------------------------------
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
                msg = ChatMessage(role=mr["role"], content=mr["content"])
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

    # ------------------------------------------------------------------
    # 从旧 JSON 文件迁移
    # ------------------------------------------------------------------
    def _maybe_migrate_json(self) -> None:
        assert self._conn is not None
        candidates = [
            self._persist_path.with_suffix(".json"),
            self._persist_path.parent / "conversations.json",
        ]
        for json_path in candidates:
            if json_path and json_path.exists():
                try:
                    with json_path.open("r", encoding="utf-8") as f:
                        raw = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue

                items = raw.get("conversations") if isinstance(raw, dict) else raw
                if not isinstance(items, list):
                    continue

                migrated = 0
                for data in items:
                    try:
                        sid = data.get("session_id") or uuid.uuid4().hex
                        title = data.get("title", "新对话")
                        sp = data.get("system_prompt")
                        ca = data.get("created_at") or int(time.time())
                        ua = data.get("updated_at") or int(time.time())
                        messages = data.get("messages") or []
                    except Exception:
                        continue

                    # 已存在则跳过（幂等）
                    row = self._conn.execute(
                        "SELECT 1 FROM conversations WHERE session_id = ?",
                        (sid,),
                    ).fetchone()
                    if row:
                        continue

                    self._conn.execute(
                        "INSERT INTO conversations(session_id, title, system_prompt, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (sid, title, sp, ca, ua),
                    )
                    for m in messages:
                        if not isinstance(m, dict):
                            continue
                        self._conn.execute(
                            "INSERT INTO messages(session_id, role, content, created_at) "
                            "VALUES (?, ?, ?, ?)",
                            (sid, m.get("role", "user"), m.get("content", ""), int(time.time())),
                        )
                    migrated += 1

                if migrated:
                    # 迁移完成后把原文件改名备份
                    try:
                        json_path.replace(json_path.with_suffix(".json.bak"))
                    except OSError:
                        pass

                return  # 只迁移一次第一个文件

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------
    def create(
        self,
        title: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Conversation:
        with self._lock:
            conv = Conversation(title=title or "新对话", system_prompt=system_prompt)
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
                # 依赖 ON DELETE CASCADE 自动清 messages
                self._conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,),
                )
            return True

    def append_message(self, session_id: str, message: ChatMessage) -> Optional[Conversation]:
        with self._lock:
            conv = self._store.get(session_id)
            if conv is None:
                return None
            conv.append(message)
            if self._conn is not None:
                self._conn.execute(
                    "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, message.role, message.content, conv.updated_at),
                )
                self._touch_updated_at(session_id)
            return conv

    def extend_messages(self, session_id: str, messages: List[ChatMessage]) -> Optional[Conversation]:
        with self._lock:
            conv = self._store.get(session_id)
            if conv is None:
                return None
            conv.extend(messages)
            if self._conn is not None:
                now = int(time.time())
                self._conn.executemany(
                    "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    [(session_id, m.role, m.content, now) for m in messages],
                )
                self._touch_updated_at(session_id)
            return conv

    # ------------------------------------------------------------------
    # 只读
    # ------------------------------------------------------------------
    def get(self, session_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._store.get(session_id)

    def list_all(self) -> List[Conversation]:
        with self._lock:
            items = [self._store[sid] for sid in self._order if sid in self._store]
            items.sort(key=lambda c: c.updated_at, reverse=True)
            return items
