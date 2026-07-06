"""Application Profile —— 配置驱动的应用定义。

一个 Application Profile = 一个独立的 AI 应用：
    - system_prompt: 系统 Prompt 标识（对应 prompts/xxx.txt）
    - tools: 该应用可用的工具列表

做一个新应用的流程：
    1. 在 prompts/ 目录放一个 .txt 文件写系统 Prompt
    2. 写一行配置:
           code_review = ApplicationProfile(
               name="code_review",
               system_prompt="code_review",     # 对应 prompts/code_review.txt
               tools=["calculator", "datetime", "text_stats"],
           )
    3. 在 app.py 注册:
           applications = {"default": default_profile, "code_review": code_review}
    4. 完成 —— 不需要改任何核心代码

这就是样板项目的价值：做应用 = 写配置，不是改框架。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ai_agent.tools.base import BaseTool, ToolRegistry


@dataclass
class ApplicationProfile:
    """一个 AI 应用的完整定义。

    Attributes:
        name:         应用名（在路由、日志、会话中使用）
        system_prompt: 系统 Prompt。可以是：
                        - 一个 prompts/ 目录下的文件名（.txt 可省略）
                        - 直接传字符串（包含换行则视为 inline prompt）
        tools:        工具名列表（必须是 ToolRegistry 中已注册的工具）
        max_iterations: 最大循环次数（防止死循环）
        description:  人类可读的描述，用于日志/UI 展示
    """

    name: str
    system_prompt: str = "agent_system"
    tools: List[str] = field(default_factory=list)
    max_iterations: int = 5
    description: str = ""

    def resolve_system_prompt(self) -> str:
        """把 system_prompt 字段解析成最终文本。"""
        if "\n" in self.system_prompt.strip():
            return self.system_prompt.strip()

        from ai_agent.prompts.prompt_loader import load_prompt

        return load_prompt(self.system_prompt, default=load_prompt("defaults/system", default="你是一个智能助手"))

    def resolve_tools(self, registry: ToolRegistry) -> List[BaseTool]:
        """根据 tools 名字列表从 registry 拿到实际工具对象。"""
        if not self.tools:
            return []

        if self.tools == ["*"]:
            return list(registry.all())

        resolved: List[BaseTool] = []
        missing: List[str] = []
        for tool_name in self.tools:
            tool = registry.get(tool_name)
            if tool is None:
                missing.append(tool_name)
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
        """快速创建一个 Agent 应用（带工具）。"""
        return cls(
            name=name,
            system_prompt=system_prompt,
            tools=tools or ["*"],
            max_iterations=max_iterations,
            description=f"Agent 应用 {name}",
        )
