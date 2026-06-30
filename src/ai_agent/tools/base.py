"""Tool 接口定义。

每个 Tool 提供：
- name:        工具名（英文、唯一、可读）
- description: 工具描述（给 LLM 看的）
- args_schema: JSON Schema 格式的参数定义（给 LLM 看的 + 运行时校验）
- run(args):   实际执行

设计原则：
1. 工具本身**不感知 LLM**，只管"输入 dict → 输出 str/dict"
2. 异常由 run() 抛出，AgentRuntime 负责捕获并转化为"工具失败"的文本
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """所有工具的基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名。唯一、简短、英文。例如 'calculator'。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """给 LLM 看的工具说明。说清楚什么时候用。"""

    @property
    @abstractmethod
    def args_schema(self) -> Dict[str, Any]:
        """JSON Schema 描述工具参数。例如：
        {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，支持 + - * / ** ()"}
            },
            "required": ["expression"]
        }
        """

    @abstractmethod
    def run(self, args: Dict[str, Any]) -> Any:
        """执行工具。返回值最好是 str 或可 JSON 序列化的对象。"""

    # ---------- 便捷方法：把工具信息序列化成 LLM 可读的描述 ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
        }

    def to_description_text(self) -> str:
        """用简洁文本描述工具，给 Planner Prompt 用。"""
        schema = json.dumps(self.args_schema, ensure_ascii=False, indent=2)
        return (
            f"- 【{self.name}】{self.description}\n"
            f"  参数 JSON Schema：\n{schema}"
        )


class ToolRegistry:
    """工具容器：注册 + 按名查找。"""

    def __init__(self, tools: List[BaseTool] | None = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具名冲突：{tool.name} 已注册")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def to_description_text(self) -> str:
        if not self._tools:
            return "（无可用工具）"
        return "\n\n".join(t.to_description_text() for t in self._tools.values())
