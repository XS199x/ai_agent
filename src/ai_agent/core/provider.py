"""Provider 基类和工具提供者接口。

设计原则：
1. Provider 只提供资源，不执行
2. 所有 Provider 共享 setup/teardown/health 生命周期
3. 工具相关的 Provider 在此定义
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ai_agent.models.action import Action
from ai_agent.tools.base import BaseTool


class Provider(ABC):
    """所有 Provider 的基类。"""

    @abstractmethod
    async def setup(self) -> None:
        """初始化 Provider。"""
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """清理资源。"""
        pass

    @abstractmethod
    async def health(self) -> bool:
        """健康检查。"""
        pass


class ToolProvider(Provider):
    """工具提供者接口。"""

    @abstractmethod
    def get_tool(self, name: str) -> Optional["BaseTool"]:
        """根据名称获取工具。"""
        pass

    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具（包含名称、描述、参数）。"""
        pass

    @abstractmethod
    def as_actions(self) -> List[Action]:
        """把工具列表转换成 Action 列表。"""
        pass