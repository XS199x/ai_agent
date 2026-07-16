import asyncio
import json
from typing import Any, List, Optional

from ai_agent.core.policy import CancellationToken
from ai_agent.core.provider import ToolProvider
from ai_agent.llm.base import BaseLLM
from ai_agent.models.action import Action, answer_action, error_action, tool_action
from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import AgentContext


class Planner:
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
        self._has_native_tools = hasattr(llm, "chat_with_tools")
        self._system_prompt = system_prompt or load_prompt(
            "agent_system", default=self.DEFAULT_FALLBACK
        )

    async def plan(
        self, context: AgentContext, token: Optional[CancellationToken] = None
    ) -> Action:
        token = token or CancellationToken()

        try:
            token.raise_if_cancelled()
            messages = self._build_messages(context)

            if self._has_native_tools:
                tool_objects = self._resolve_tools(context, token)
                if tool_objects:
                    response = await self._llm.chat_with_tools(
                        messages=messages, tools=tool_objects
                    )
                    return self._parse_tool_response(response)

            response = await self._llm.chat(messages=messages)
            return self._parse_text_response(response)

        except asyncio.CancelledError:
            return error_action("Planner 已取消")
        except Exception as e:
            return error_action(f"Planner 决策失败：{str(e)}")

    def _build_messages(self, context: AgentContext) -> List[ChatMessage]:
        system_content = self._system_prompt
        snippets = (getattr(context, "system_prompt_snippets", "") or "").strip()
        if snippets:
            system_content = f"{system_content.rstrip()}\n\n{snippets}"

        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=system_content)
        ]

        last_has_tool_calls = False
        for msg in context.conversation:
            if msg.role == "tool":
                if last_has_tool_calls:
                    messages.append(msg)
                last_has_tool_calls = False
            else:
                messages.append(msg)
                last_has_tool_calls = bool(msg.tool_calls)

        return messages

    def _resolve_tools(
        self, context: AgentContext, token: CancellationToken
    ) -> List[Any]:
        if not self._tool_provider:
            return []

        available_names = {
            action.name for action in context.available_actions if hasattr(action, "name")
        }
        if not available_names:
            return []

        result = []
        for name in available_names:
            token.raise_if_cancelled()
            tool = self._tool_provider.get_tool(name)
            if tool:
                result.append(tool)
        return result

    def _parse_tool_response(self, response: Any) -> Action:
        try:
            if hasattr(response, "choices") and response.choices:
                message = response.choices[0].message

                if hasattr(message, "tool_calls") and message.tool_calls:
                    tc = message.tool_calls[0]
                    func_name = getattr(tc.function, "name", "")
                    func_args_str = getattr(tc.function, "arguments", "{}")
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

            return answer_action(content="无法理解您的请求", thought="LLM 返回格式不明确")
        except Exception as e:
            return error_action(f"解析 LLM 返回失败：{str(e)}")

    def _parse_text_response(self, response: Any) -> Action:
        try:
            if hasattr(response, "choices") and response.choices:
                message = response.choices[0].message
                content = getattr(message, "content", "")
                if content:
                    return answer_action(
                        content=content, thought="LLM 直接回答用户"
                    )
            return answer_action(content="无法理解您的请求", thought="LLM 返回格式不明确")
        except Exception as e:
            return error_action(f"解析 LLM 返回失败：{str(e)}")
