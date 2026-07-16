"""对话领域模型：纯数据结构，不包含持久化逻辑。"""

import time
import uuid
from typing import List, Optional

from ai_agent.models.chat import ChatMessage


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


__all__ = ["Conversation"]
