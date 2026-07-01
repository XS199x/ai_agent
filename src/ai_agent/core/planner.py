"""Planner：分析用户意图，决定是否/调用哪个工具。

实现策略：
- 让 LLM 以**严格 JSON** 输出决策：{"use_tool": true/false, "tool": "...", "args": {...}, "reason": "..."}
- Prompt 里明确写出所有可用工具的 name / description / args_schema
- 解析失败时：fallback 到 use_tool = false，不会崩溃

这样设计的好处：
1. 不依赖 Function Calling 能力，对任何模型都可用
2. Prompt 中可以写更多上下文（如"用户历史里有提到计算器"）
3. 失败不会影响主线——聊天气味的仍然可以聊
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.ai_agent.llm.base import BaseLLM
from src.ai_agent.models.chat import ChatMessage
from src.ai_agent.tools.base import ToolRegistry

_PLANNER_SYSTEM_PROMPT = """你是一个**工具选择器**（Tool Selector）。

你的任务：只分析用户的最新输入，判断是否需要调用外部工具来获取准确答案。

**可用工具列表**：

{tools_description}

**输出格式**（必须是严格的 JSON，不要加任何解释、引号、Markdown、代码块标记）：

情形 1：需要工具
{{
  "use_tool": true,
  "tool": "工具名（必须是上面列表中的一个）",
  "args": {{ 参数名: 参数值 }},
  "reason": "简短说明为什么需要这个工具"
}}

情形 2：不需要工具（闲聊 / 常识 / 写作 / 翻译 / LLM 自己能回答的问题）
{{
  "use_tool": false,
  "reason": "说明为什么不需要工具"
}}

**严格约束**：
- 必须只输出 JSON，不要输出任何其他文字
- 不要用 ```json 或 ``` 代码块
- 不要在 JSON 前后加解释文字
- 字段顺序必须严格如上
- "args" 必须是 dict，不能是字符串
"""


@dataclass
class PlannerDecision:
    """Planner 的决策结果。"""

    use_tool: bool
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    reason: str = ""
    raw_response: str = ""  # 方便调试

    @classmethod
    def no_tool(cls, reason: str = "") -> "PlannerDecision":
        return cls(use_tool=False, reason=reason)

    @classmethod
    def from_json(cls, data: Dict[str, Any], raw: str = "") -> "PlannerDecision":
        use_tool = bool(data.get("use_tool", False))
        if not use_tool:
            return cls(
                use_tool=False, reason=str(data.get("reason", "")), raw_response=raw
            )
        return cls(
            use_tool=True,
            tool=str(data.get("tool") or ""),
            args=data.get("args") if isinstance(data.get("args"), dict) else {},
            reason=str(data.get("reason", "")),
            raw_response=raw,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "use_tool": self.use_tool,
            "tool": self.tool,
            "args": self.args,
            "reason": self.reason,
        }


class Planner:
    def __init__(self, llm: BaseLLM, registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = registry

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """从 LLM 可能的自由输出里提取 JSON 字符串。

        策略：
        1. 如果看起来就是完整 JSON，直接返回
        2. 如果包裹在 ```json ... ``` 里，取中间
        3. 找第一个 { 和最后一个 }
        """
        text = text.strip()
        if not text:
            return None
        # 纯 JSON（首尾就是大括号）
        if text.startswith("{") and text.endswith("}"):
            return text
        # ```json ... ``` 或 ``` ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return m.group(1)
        # 找第一个 { 和最后一个 }
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first : last + 1]
        return None

    async def plan(self, user_messages: List[ChatMessage]) -> PlannerDecision:
        """对用户问题做决策。注意：看完整上下文（用户消息 + 之前的工具结果），不只看最后一条。"""
        # 没有工具就直接返回
        if len(self.registry) == 0:
            return PlannerDecision.no_tool("没有注册任何工具")

        if not user_messages:
            return PlannerDecision.no_tool("空输入")

        tools_description = self.registry.to_description_text()
        system_text = _PLANNER_SYSTEM_PROMPT.format(tools_description=tools_description)

        # 把完整上下文转给 Planner（用户消息 + 已产生的工具结果消息）
        # 这样 Planner 可以理解多轮对话（如：用户先问数学，再追问"再算一个"）
        planner_messages: List[ChatMessage] = [
            ChatMessage(role="system", content=system_text)
        ]
        planner_messages.extend(user_messages)

        try:
            resp = await self.llm.chat(planner_messages)
        except Exception as e:
            return PlannerDecision.no_tool(f"Planner 调用 LLM 失败：{e}")

        if not resp.choices:
            return PlannerDecision.no_tool("LLM 返回为空")
        raw = resp.choices[0].message.content or ""

        json_text = self._extract_json(raw)
        if not json_text:
            return PlannerDecision(
                use_tool=False,
                reason=f"LLM 没有输出可解析的 JSON，raw={raw[:120]!r}",
                raw_response=raw,
            )

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            return PlannerDecision(
                use_tool=False,
                reason=f"JSON 解析失败：{e}。raw={json_text[:120]!r}",
                raw_response=raw,
            )

        decision = PlannerDecision.from_json(data, raw=raw)

        # 基本校验：如果要走工具，确认 tool 存在且 args 是 dict
        if decision.use_tool:
            if not decision.tool:
                return PlannerDecision(
                    use_tool=False,
                    reason="Planner 返回了 use_tool=true 但没写 tool 名",
                    raw_response=raw,
                )
            if self.registry.get(decision.tool) is None:
                return PlannerDecision(
                    use_tool=False,
                    reason=f"Planner 选择了不存在的工具：{decision.tool!r}",
                    raw_response=raw,
                )
            if not isinstance(decision.args, dict):
                return PlannerDecision(
                    use_tool=False,
                    reason=f"Planner 的 args 不是 dict：{type(decision.args).__name__}",
                    raw_response=raw,
                )

        return decision
