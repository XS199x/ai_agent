from typing import AsyncGenerator, List

from src.ai_agent.config import AgentConfig
from src.ai_agent.config import config as global_config
from src.ai_agent.core.loop import AgentLoop
from src.ai_agent.llm.base import BaseLLM
from src.ai_agent.models.chat import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
)


class Agent:
    def __init__(self, llm: BaseLLM, agent_config: AgentConfig = None) -> None:
        self.llm = llm
        self.config = agent_config or global_config.agent
        self.loop = AgentLoop(llm, self.config)

    async def run(self, messages: List[ChatMessage]) -> ChatCompletionResponse:
        return await self.loop.run(messages)

    def run_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        async def _stream() -> AsyncGenerator[ChatCompletionChunk, None]:
            async for chunk in self.loop.run_stream(messages):
                yield chunk

        return _stream()

    async def chat(self, user_message: str) -> ChatCompletionResponse:
        messages = [ChatMessage(role="user", content=user_message)]
        return await self.run(messages)

    def chat_stream(
        self, user_message: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        messages = [ChatMessage(role="user", content=user_message)]
        return self.loop.run_stream(messages)
