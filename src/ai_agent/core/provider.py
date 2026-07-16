from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from ai_agent.models.action import Action
from ai_agent.tools.base import BaseTool


class Provider(ABC):
    async def setup(self) -> None: ...
    async def teardown(self) -> None: ...
    async def health(self) -> bool:
        return True


class ToolProvider(Provider):
    @abstractmethod
    def get_tool(self, name: str) -> Optional["BaseTool"]: ...

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def as_actions(self) -> List[Action]: ...


class CompositeToolProvider(ToolProvider):
    def __init__(self, providers: Iterable[ToolProvider]):
        self._providers: List[ToolProvider] = [p for p in providers if p is not None]

    async def setup(self) -> None:
        errors: List[str] = []
        for p in self._providers:
            try:
                await p.setup()
            except Exception as e:
                errors.append(f"{type(p).__name__}.setup failed: {e}")
        if errors and not self._providers:
            raise RuntimeError("; ".join(errors))

    async def teardown(self) -> None:
        for p in self._providers:
            try:
                await p.teardown()
            except Exception:
                pass

    async def health(self) -> bool:
        if not self._providers:
            return False
        for p in self._providers:
            try:
                if await p.health():
                    return True
            except Exception:
                pass
        return False

    def get_tool(self, name: str) -> Optional["BaseTool"]:
        for p in self._providers:
            tool = p.get_tool(name)
            if tool is not None:
                return tool
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        out: List[Dict[str, Any]] = []
        for p in self._providers:
            for desc in p.list_tools():
                name = desc.get("name") if isinstance(desc, dict) else None
                if not name or name in seen:
                    continue
                seen.add(name)
                out.append(desc)
        return out

    def as_actions(self) -> List[Action]:
        seen: set[str] = set()
        out: List[Action] = []
        for p in self._providers:
            for act in p.as_actions():
                nm = getattr(act, "name", None)
                if not nm or nm in seen:
                    continue
                seen.add(nm)
                out.append(act)
        return out
