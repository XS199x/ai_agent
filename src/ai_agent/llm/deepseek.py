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


class DeepSeekLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def format_messages(
        self, messages: List[ChatMessage]
    ) -> List[ChatCompletionMessageParam]:
        result = []
        for msg in messages:
            base_item = {"role": msg.role, "content": msg.content}
            if msg.role == "tool":
                if msg.tool_call_id:
                    base_item["tool_call_id"] = msg.tool_call_id
            else:
                if msg.name:
                    base_item["name"] = msg.name
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
                        role=choice.message.role,
                        content=choice.message.content or "",
                        name=choice.message.name,
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
        self, messages: List[ChatMessage]
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
                delta_content = choice.delta.content or ""
                delta_role = choice.delta.role or "assistant"
                choices.append(
                    ChatCompletionChunkChoice(
                        index=choice.index,
                        delta=ChatMessage(
                            role=delta_role,
                            content=delta_content,
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
