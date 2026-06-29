from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from src.ai_agent.config import LLMConfig
from src.ai_agent.models.chat import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
)


class BaseLLM(ABC):
    config: LLMConfig

    @abstractmethod
    async def chat(self, messages: List[ChatMessage]) -> ChatCompletionResponse:
        pass

    @abstractmethod
    def chat_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        pass

    @abstractmethod
    def format_messages(self, messages: List[ChatMessage]) -> List[dict]:
        pass
