from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ai_agent.tools.base import BaseTool, ToolRegistry


@dataclass
class ApplicationProfile:
    name: str
    system_prompt: str = "agent_system"
    tools: List[str] = field(default_factory=list)
    max_iterations: int = 5
    description: str = ""

    def resolve_system_prompt(self) -> str:
        if "\n" in self.system_prompt.strip():
            return self.system_prompt.strip()
        from ai_agent.prompts.prompt_loader import load_prompt
        return load_prompt(
            self.system_prompt,
            default=load_prompt("defaults/system", default="你是一个智能助手"),
        )

    def resolve_tools(self, registry: ToolRegistry) -> List[BaseTool]:
        if not self.tools:
            return []
        if self.tools == ["*"]:
            return list(registry.all())

        resolved: List[BaseTool] = []
        missing: List[str] = []
        for name in self.tools:
            tool = registry.get(name)
            if tool is None:
                missing.append(name)
            else:
                resolved.append(tool)

        if missing:
            raise ValueError(
                f"Application '{self.name}' 配置了不存在的工具: {missing}。"
                f"可用工具: {[t.name for t in registry.all()]}"
            )
        return resolved

    @classmethod
    def agent_app(
        cls,
        name: str,
        system_prompt: str = "agent_system",
        tools: Optional[List[str]] = None,
        max_iterations: int = 5,
    ) -> "ApplicationProfile":
        return cls(
            name=name,
            system_prompt=system_prompt,
            tools=tools or ["*"],
            max_iterations=max_iterations,
            description=f"Agent 应用 {name}",
        )
