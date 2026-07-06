"""Tool 接口定义。

每个 Tool 提供：
- name:        工具名（英文、唯一、可读）
- description: 工具描述（给 LLM 看的）
- args_schema: JSON Schema 格式的参数定义（给 LLM 看的 + 运行时校验）
- run(args):   实际执行（由子类实现）
- execute(args): 带超时 + 异常隔离 + 标准化返回的包装方法（AgentRuntime 调用这个）

设计原则：
1. 工具本身**不感知 LLM**，只管"输入 dict → 输出 str/dict"
2. run() 可以抛异常；execute() 负责捕获异常并统一返回 ToolResult
3. 通用层**不知道**任何具体协议（function calling 等），协议适配在 LLM Adapter 里
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------- 工具执行结果 ----------


@dataclass
class ToolResult:
    """工具执行的标准化返回值。AgentRuntime 只感知这个类型。"""

    success: bool
    output: str = ""  # 成功时的可读输出（会塞进上下文中给 LLM 看）
    error: Optional[str] = (
        None  # 失败时的错误说明（也会塞进上下文给 LLM，让它决定是否重试）
    )
    duration_ms: int = 0  # 实际执行耗时
    raw: Any = None  # 原始返回值（可选，方便调试/事件记录）

    def display_text(self) -> str:
        """给 LLM 看的文本表示。"""
        if self.success:
            return self.output or ""
        return f"工具调用失败：{self.error or '未知错误'}"


# ---------- 参数 Schema 校验 ----------


def validate_args(args: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    """用 Python 内置逻辑做轻量级 JSON Schema 校验。

    不依赖第三方库，只校验我们真正关心的字段。
    返回 None 表示 OK，返回 str 表示错误原因（会回传给 LLM 让它重试）。
    """
    if not isinstance(args, dict):
        return f"参数必须是 object/dict，实际是 {type(args).__name__}"

    required: List[str] = schema.get("required", []) or []
    for key in required:
        if key not in args:
            return f"缺少必填字段 {key!r}"

    properties: Dict[str, Any] = schema.get("properties", {}) or {}
    for key, value in args.items():
        prop_schema = properties.get(key)
        if prop_schema is None:
            return f"未知字段 {key!r}（不在 Schema 的 properties 中）"
        expected_type = prop_schema.get("type")
        if expected_type and not _type_matches(value, expected_type):
            return f"字段 {key!r} 类型错误：期望 {expected_type}，实际是 {type(value).__name__}"
    return None


def _type_matches(value: Any, expected_type: str) -> bool:
    mapping = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    py_type = mapping.get(expected_type)
    if py_type is None:
        return True  # 未知类型，放行
    # bool 是 int 的子类——但 "integer" 校验时 true/false 不应通过
    if expected_type == "integer" and isinstance(value, bool):
        return False
    if expected_type == "number" and isinstance(value, bool):
        return False
    return isinstance(value, py_type)


# ---------- 基类 ----------


class BaseTool(ABC):
    """所有工具的基类。"""

    @property
    def timeout(self) -> float:
        """单次执行超时（秒）。子类可以覆盖。"""
        return 10.0

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
        """实际执行逻辑。返回 str 或可 JSON 序列化对象。
        可以抛异常 —— execute() 会捕获。"""

    # ---------- 对外统一入口：带超时 + 异常隔离 ----------

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """AgentRuntime 应该调用这个方法。会做：
        1. Schema 校验（参数不合法 → 直接失败返回 ToolResult，不执行）
        2. 计时
        3. 执行 run() + 捕获异常
        4. 标准化返回 ToolResult
        """
        schema_error = validate_args(args or {}, self.args_schema)
        if schema_error is not None:
            return ToolResult(
                success=False,
                error=f"参数校验失败：{schema_error}。"
                f"（Schema: {json.dumps(self.args_schema, ensure_ascii=False)}）",
                duration_ms=0,
            )

        start = time.monotonic()
        try:
            raw_output = self.run(args or {})
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=duration_ms,
                raw=None,
            )

        duration_ms = int((time.monotonic() - start) * 1000)
        # 将输出统一为可读字符串
        if isinstance(raw_output, str):
            output_text = raw_output
        else:
            try:
                output_text = json.dumps(raw_output, ensure_ascii=False, indent=2)
            except Exception:
                output_text = str(raw_output)

        return ToolResult(
            success=True,
            output=output_text,
            duration_ms=duration_ms,
            raw=raw_output,
        )

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
        return f"- 【{self.name}】{self.description}\n  参数 JSON Schema：\n{schema}"


# ---------- 工具注册中心 ----------


class ToolRegistry:
    """工具容器：注册 + 按名查找。"""

    def __init__(self, tools: List[BaseTool] | None = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            self.register_many(tools)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具名冲突：{tool.name} 已注册")
        self._tools[tool.name] = tool

    def register_many(self, tools: List[BaseTool]) -> None:
        for t in tools:
            self.register(t)

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


# ---------- Tool 工厂：用一行代码把 callable 包装成 BaseTool ----------

# 简化版参数 Schema 定义，方便工厂调用
ToolParamSpec = Dict[str, Any]  # {param_name: {"type": "string", "description": "..."}}


def make_tool(
    name: str,
    description: str,
    params: ToolParamSpec,
    required: Optional[List[str]] = None,
    timeout: float = 10.0,
):
    """把一个 callable 函数包装成 BaseTool，避免写样板类。

    用法示例：

        @make_tool(
            name="read_file",
            description="读取本地文件的内容，用于分析或回答问题",
            params={"path": {"type": "string", "description": "文件路径，相对项目根目录"}},
            required=["path"],
        )
        def read_file(args: Dict[str, Any]) -> str:
            return Path(args["path"]).read_text(encoding="utf-8")

    也可以直接调用返回值手动注册：
        registry.register(read_file)
    """

    required_list = required or []
    args_schema = {
        "type": "object",
        "properties": params,
        "required": required_list,
    }

    def decorator(func):
        class _GenericTool(BaseTool):
            def __init__(self) -> None:
                self._name = name
                self._description = description
                self._args_schema = args_schema
                self._timeout = timeout
                self._func = func

            @property
            def name(self):
                return self._name

            @property
            def description(self):
                return self._description

            @property
            def args_schema(self):
                return self._args_schema

            @property
            def timeout(self):
                return self._timeout

            def run(self, args: Dict[str, Any]) -> str:
                return self._func(args)

        _GenericTool.__name__ = f"Tool_{name}"
        return _GenericTool()

    return decorator
