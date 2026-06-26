from datetime import datetime
from typing import List, Optional

from src.ai_agent.models.chat import ChatMessage


class MemoryEntry:
    def __init__(self, content: str, timestamp: Optional[datetime] = None):
        self.content = content
        self.timestamp = timestamp or datetime.now()


class Memory:
    def __init__(self) -> None:
        self.short_term: List[MemoryEntry] = []
        self.long_term: List[MemoryEntry] = []
        self.max_short_term = 100

    def add_short_term(self, content: str) -> None:
        self.short_term.append(MemoryEntry(content))
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term :]

    def add_long_term(self, content: str) -> None:
        self.long_term.append(MemoryEntry(content))

    def retrieve_recent(self, count: int = 10) -> List[str]:
        return [entry.content for entry in self.short_term[-count:]]

    def retrieve_all(self) -> List[str]:
        all_entries = self.long_term + self.short_term
        return [entry.content for entry in all_entries]

    def clear_short_term(self) -> None:
        self.short_term = []

    def clear_all(self) -> None:
        self.short_term = []
        self.long_term = []
