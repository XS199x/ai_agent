from typing import AsyncGenerator, List, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.ai_agent.config import LLMConfig
from src.ai_agent.llm.base import BaseLLM
from src.ai_agent.models.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionResponse,
    ChatMessage,
)

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
            base_item = {"role": msg.role, "content": msg.content}
            if msg.name:
                base_item["name"] = msg.name
            if msg.role == "tool":
                if msg.tool_call_id:
                    base_item["tool_call_id"] = msg.tool_call_id
            else:
                if msg.tool_call_id:
                    base_item["tool_call_id"] = msg.tool_call_id
            item = cast(ChatCompletionMessageParam, base_item)
            result.append(item)
        return result

    async def chat(self, messages: List[ChatMessage]) -> ChatCompletionResponse:
        formatted_messages = self.format_messages(messages)
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=formatted_messages,
            temperature=self.config.temperature,
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
