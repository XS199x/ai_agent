"""LocalToolProvider：本地工具提供者。

把现有的 ToolRegistry + BaseTool 体系适配到新的 Provider 接口。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_agent.core.executor import ToolProvider
from ai_agent.models.action import Action, ToolAction, tool_action
from ai_agent.tools.base import BaseTool, ToolRegistry


class LocalToolProvider(ToolProvider):
    """基于本地 ToolRegistry 的工具提供者。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def setup(self) -> None:
        """初始化（本地工具无需特殊初始化）。"""
        pass

    async def teardown(self) -> None:
        """清理（本地工具无需特殊清理）。"""
        pass

    async def health(self) -> bool:
        """健康检查。"""
        return True

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具。"""
        return self._registry.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具。"""
        tools = []
        for tool in self._registry.all():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema,
            })
        return tools

    def as_actions(self) -> List[Action]:
        """把工具列表转换成 Action 列表。"""
        actions = []
        for tool in self._registry.all():
            actions.append(tool_action(name=tool.name))
        return actions
