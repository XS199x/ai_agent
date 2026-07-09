"""Planner：决策层。

设计原则：
1. Planner 永远不执行
2. Planner 永远不查数据库
3. Planner 永远不访问 MCP
4. Planner 只负责：下一步应该干什么？

Planner 的职责：
- 接收 AgentContext
- 构建完整的 Prompt（system_prompt + conversation）
- 调用 LLM 获取决策
- 解析 LLM 返回，输出统一的 Action

Planner 和 Executor 通过 Action 解耦。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ai_agent.core.provider import ToolProvider
from ai_agent.llm.base import BaseLLM
from ai_agent.models.action import (
    Action,
    AnswerAction,
    ErrorAction,
    answer_action,
    error_action,
    tool_action,
)
from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import AgentContext


class Planner(ABC):
    """Planner 接口。"""

    @abstractmethod
    async def plan(self, context: AgentContext) -> Action:
        """根据上下文做出决策，返回下一步的 Action。"""
        pass


class LLMPlanner(Planner):
    """基于 LLM 的 Planner。

    使用函数调用（function calling）来获取结构化的 Action 决策。
    优先使用 LLM 的原生 function calling，如果不支持则降级为文本解析。

    系统 Prompt 从 prompts/agent_system.txt 加载（修改文件不用改代码）。
    """

    DEFAULT_FALLBACK = "你是一个智能助手。"

    def __init__(
        self,
        llm: BaseLLM,
        tool_provider: Optional[ToolProvider] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        from ai_agent.prompts.prompt_loader import load_prompt

        self._llm = llm
        self._tool_provider = tool_provider
        self._has_native_tool_calling = hasattr(llm, "chat_with_tools")

        if system_prompt is not None:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = load_prompt(
                "agent_system", default=self.DEFAULT_FALLBACK
            )

    async def plan(self, context: AgentContext) -> Action:
        messages = self._build_messages(context)

        try:
            if self._has_native_tool_calling:
                return await self._plan_with_native_tools(messages, context)
            else:
                return await self._plan_with_fallback(messages, context)
        except Exception as e:
            return error_action(f"Planner 决策失败：{str(e)}")

    def _build_messages(self, context: AgentContext) -> List[ChatMessage]:
        """构建传给 LLM 的消息列表。"""
        messages: List[ChatMessage] = []

        messages.append(ChatMessage(role="system", content=self._system_prompt))

        messages.extend(context.conversation)

        return messages

    async def _plan_with_native_tools(
        self, messages: List[ChatMessage], context: AgentContext
    ) -> Action:
        """使用原生 function calling。"""
        if self._tool_provider is None:
            return await self._plan_with_fallback(messages, context)

        real_tools = self._tool_provider.list_tools()
        tool_objects = []
        for tool_info in real_tools:
            tool_obj = self._tool_provider.get_tool(tool_info["name"])
            if tool_obj:
                tool_objects.append(tool_obj)

        if not tool_objects:
            return await self._plan_with_fallback(messages, context)

        response = await self._llm.chat_with_tools(
            messages=messages,
            tools=tool_objects,
        )

        return self._parse_tool_response(response)

    async def _plan_with_fallback(
        self, messages: List[ChatMessage], context: AgentContext
    ) -> Action:
        """降级方案：使用普通 chat，通过 Prompt 引导输出 JSON。"""
        response = await self._llm.chat(messages=messages)
        return self._parse_text_response(response)

    def _parse_tool_response(self, response: Any) -> Action:
        """解析 function calling 响应。"""
        try:
            if hasattr(response, "choices") and response.choices:
                message = response.choices[0].message

                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    func_name = getattr(tool_call.function, "name", "")
                    func_args_str = getattr(tool_call.function, "arguments", "{}")

                    try:
                        func_args = json.loads(func_args_str)
                    except Exception:
                        func_args = {}

                    return tool_action(
                        name=func_name,
                        args=func_args,
                        thought=f"LLM 选择调用工具 {func_name}",
                    )

                if hasattr(message, "content") and message.content:
                    return answer_action(
                        content=message.content,
                        thought="LLM 认为可以直接回答用户",
                    )

            return answer_action(
                content="无法理解您的请求",
                thought="LLM 返回格式不明确",
            )

        except Exception as e:
            return error_action(f"解析 LLM 返回失败：{str(e)}")

    def _parse_text_response(self, response: Any) -> Action:
        """解析文本响应（降级方案）。"""
        try:
            if hasattr(response, "choices") and response.choices:
                message = response.choices[0].message
                content = getattr(message, "content", "")

                if content:
                    return answer_action(
                        content=content,
                        thought="LLM 直接回答用户",
                    )

            return answer_action(
                content="无法理解您的请求",
                thought="LLM 返回格式不明确",
            )

        except Exception as e:
            return error_action(f"解析 LLM 返回失败：{str(e)}")
