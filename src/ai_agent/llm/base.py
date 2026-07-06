from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from ai_agent.config import LLMConfig
from ai_agent.models.chat import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
)


class BaseLLM(ABC):
    config: LLMConfig

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
    ) -> ChatCompletionResponse:
        pass

    @abstractmethod
    def chat_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        pass

    @abstractmethod
    def format_messages(self, messages: List[ChatMessage]) -> List[dict]:
        pass
