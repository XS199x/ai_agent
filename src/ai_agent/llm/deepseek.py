from typing import Any, AsyncGenerator, Dict, List, Optional, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ai_agent.config import LLMConfig
from ai_agent.llm.base import BaseLLM
from ai_agent.models.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
)
from ai_agent.tools.base import BaseTool

_VALID_ROLES = {"system", "user", "assistant", "tool", "developer"}


class DeepSeekLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        if not config.api_key or not config.api_key.strip():
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。请在项目根目录创建 .env 文件 "
                "(从 .env.example 复制) 并填入你的 API Key，或设置环境变量 DEEPSEEK_API_KEY。"
            )
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def format_messages(
        self, messages: List[ChatMessage]
    ) -> List[ChatCompletionMessageParam]:
        result = []
        for msg in messages:
            if msg.role not in _VALID_ROLES:
                raise ValueError(
                    f"角色 '{msg.role}' 对 DeepSeek 无效，有效角色: {sorted(_VALID_ROLES)}"
                )
            if msg.role == "tool" and not msg.tool_call_id:
                prefix = (
                    f"[工具 {msg.name or 'unknown'} 结果] "
                    if msg.name
                    else "[工具结果] "
                )
                base_item = {"role": "assistant", "content": prefix + (msg.content or "")}
            else:
                base_item = {"role": msg.role}
                if msg.content:
                    base_item["content"] = msg.content
                if msg.name:
                    base_item["name"] = msg.name
                if msg.tool_call_id:
                    base_item["tool_call_id"] = msg.tool_call_id
                if msg.tool_calls:
                    base_item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
            item = cast(ChatCompletionMessageParam, base_item)
            result.append(item)
        return result

    @staticmethod
    def _format_tool_schema(tool: BaseTool) -> Dict[str, Any]:
        """把通用 BaseTool 转成 DeepSeek/OpenAI function calling 需要的 schema。

        这个格式化逻辑是 DeepSeek 协议特定的，因此**内聚在 DeepSeekLLM 里**，
        不放在 BaseTool 基类中。

        输出格式：
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "执行数学计算",
                "parameters": {...}
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema,
            },
        }

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
    ) -> ChatCompletionResponse:
        formatted_messages = self.format_messages(messages)
        effective_temperature = (
            self.config.temperature if temperature is None else temperature
        )
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=formatted_messages,
            temperature=effective_temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )

        choices = []
        for choice in response.choices:
            choices.append(
                ChatCompletionChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role=choice.message.role, content=choice.message.content or ""
                    ),
                    finish_reason=choice.finish_reason,
                )
            )

        return ChatCompletionResponse(
            id=response.id,
            object=response.object,
            created=response.created,
            model=response.model,
            choices=choices,
        )

    async def chat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[BaseTool],
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ChatCompletionResponse:
        """扩展能力：走原生 function calling 调用。

        这是 DeepSeekLLM 的**特定能力**，不属于 BaseLLM 的通用接口。
        Planner 会检测有没有这个方法，有就走原生、没有就降级到 Prompt JSON。

        这里故意接收 List[BaseTool]，由** DeepSeekLLM 自己**把通用工具描述转成
        OpenAI function calling 要求的 schema 格式（内聚原则），避免 BaseTool
        被特定协议污染。
        """
        formatted_messages = self.format_messages(messages)
        effective_temperature = (
            self.config.temperature if temperature is None else temperature
        )

        tools_schema = [self._format_tool_schema(t) for t in tools]
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": effective_temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "tools": tools_schema,
        }
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**kwargs)

        choices = []
        for choice in response.choices:
            msg = choice.message
            our_message = ChatMessage(role=msg.role, content=msg.content or "")
            if msg.tool_calls:
                our_message.tool_calls = [
                    ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function=FunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                    for tc in msg.tool_calls
                ]
            choices.append(
                ChatCompletionChoice(
                    index=choice.index,
                    message=our_message,
                    finish_reason=choice.finish_reason,
                )
            )

        return ChatCompletionResponse(
            id=response.id,
            object=response.object,
            created=response.created,
            model=response.model,
            choices=choices,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:

        formatted_messages = self.format_messages(messages)

        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            choices = []

            for choice in chunk.choices:
                choices.append(
                    ChatCompletionChunkChoice(
                        index=choice.index,
                        delta=ChatMessage(
                            role=choice.delta.role or "assistant",
                            content=choice.delta.content or "",
                        ),
                        finish_reason=choice.finish_reason,
                    )
                )

            yield ChatCompletionChunk(
                id=chunk.id,
                object=chunk.object,
                created=chunk.created,
                model=chunk.model,
                choices=choices,
            )
