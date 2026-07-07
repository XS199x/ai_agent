"""基于文件的知识库提供者。

支持从目录加载 Markdown 和文本文件，提供简单的关键词检索。

设计原则：
1. 对外只暴露 KnowledgeProvider 接口
2. 内部处理文件加载、文本提取和检索
3. 支持增量加载和缓存
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from ai_agent.core.executor import KnowledgeProvider
from ai_agent.models.context import KnowledgeEntry


class FileKnowledgeProvider(KnowledgeProvider):
    """基于文件系统的知识库提供者。

    从指定目录加载所有 .md 和 .txt 文件，构建索引，支持关键词检索。
    """

    def __init__(self, knowledge_dir: Optional[str] = None) -> None:
        self._knowledge_dir = Path(knowledge_dir or "data/knowledge").resolve()
        self._entries: List[KnowledgeEntry] = []
        self._index: Dict[str, List[int]] = {}

    async def setup(self) -> None:
        self._load_files()

    async def teardown(self) -> None:
        self._entries.clear()
        self._index.clear()

    async def health(self) -> bool:
        return self._knowledge_dir.exists()

    async def retrieve(self, query: str, session_id: str) -> List[KnowledgeEntry]:
        return self._search(query)

    async def get_context(self, session_id: str, user_input: str) -> "AgentContext":
        from ai_agent.models.context import AgentContext, MemorySnapshot, RuntimeState

        return AgentContext(
            conversation=[],
            memory=MemorySnapshot(),
            knowledge=await self.retrieve(user_input, session_id),
            runtime_state=RuntimeState(session_id=session_id),
            user_input=user_input,
        )

    def _load_files(self) -> None:
        if not self._knowledge_dir.exists():
            return

        for ext in ("*.md", "*.txt"):
            for filepath in self._knowledge_dir.glob(ext):
                try:
                    content = filepath.read_text(encoding="utf-8")
                    entry = KnowledgeEntry(
                        content=content,
                        source=str(filepath.name),
                    )
                    self._entries.append(entry)
                    self._build_index(len(self._entries) - 1, content)
                except Exception:
                    continue

    def _build_index(self, entry_idx: int, content: str) -> None:
        words = self._extract_words(content)
        for word in words:
            if word not in self._index:
                self._index[word] = []
            if entry_idx not in self._index[word]:
                self._index[word].append(entry_idx)

    def _extract_words(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]", " ", text)
        words = text.split()
        return [w for w in words if len(w) >= 2]

    def _search(self, query: str) -> List[KnowledgeEntry]:
        query_words = self._extract_words(query)
        if not query_words:
            return []

        scores: Dict[int, int] = {}
        for word in query_words:
            if word in self._index:
                for idx in self._index[word]:
                    scores[idx] = scores.get(idx, 0) + 1

        results: List[KnowledgeEntry] = []
        for idx in sorted(scores.keys(), key=lambda x: -scores[x]):
            entry = self._entries[idx]
            snippet = self._extract_snippet(entry.content, query)
            results.append(
                KnowledgeEntry(
                    content=snippet,
                    source=entry.source,
                    score=scores[idx],
                )
            )

        return results[:5]

    def _extract_snippet(self, content: str, query: str) -> str:
        query_lower = query.lower()
        content_lower = content.lower()
        idx = content_lower.find(query_lower)
        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(content), idx + len(query) + 100)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(content) else ""
            return f"{prefix}{content[start:end]}{suffix}"
        lines = content.split("\n")[:3]
        return "\n".join(lines)[:200]
