from typing import AsyncGenerator, List

from src.ai_agent.config import AgentConfig
from src.ai_agent.core.memory import Memory
from src.ai_agent.core.planner import Planner
from src.ai_agent.core.state import AgentState, AgentStatus
from src.ai_agent.llm.base import BaseLLM
from src.ai_agent.models.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
)


class AgentLoop:
    def __init__(self, llm: BaseLLM, config: AgentConfig) -> None:
        self.llm = llm
        self.config = config
        self.planner = Planner()
        self.memory = Memory()

    async def run(self, messages: List[ChatMessage]) -> ChatCompletionResponse:
        state = AgentState()
        state.set_status(AgentStatus.THINKING)

        for msg in messages:
            state.add_message(msg)

        for _ in range(self.config.max_iterations):
            state.increment_iteration()

            if self.planner.should_call_tool(state.messages):
                state.set_status(AgentStatus.TOOL_CALL)
                tool_name = self.planner.select_tool(state.messages)
                tool_args = self.planner.generate_tool_args(state.messages, tool_name)
                state.add_action({"tool_name": tool_name, "tool_args": tool_args})

            else:
                state.set_status(AgentStatus.GENERATING)
                response = await self.llm.chat(state.messages)
                content = response.choices[0].message.content
                state.set_final_response(content)
                state.set_status(AgentStatus.FINISHED)
                return ChatCompletionResponse(
                    id=response.id,
                    created=response.created,
                    model=response.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=content,
                            ),
                            finish_reason="stop",
                        )
                    ],
                )

        return ChatCompletionResponse(
            id="",
            created=0,
            model="",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=state.final_response or "",
                    ),
                    finish_reason="length",
                )
            ],
        )

    def run_stream(
        self, messages: List[ChatMessage]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        async def _stream() -> AsyncGenerator[ChatCompletionChunk, None]:
            state = AgentState()
            state.set_status(AgentStatus.THINKING)

            for msg in messages:
                state.add_message(msg)

            for _ in range(self.config.max_iterations):
                state.increment_iteration()

                if self.planner.should_call_tool(state.messages):
                    state.set_status(AgentStatus.TOOL_CALL)
                    continue

                state.set_status(AgentStatus.GENERATING)
                full_content = ""
                async for chunk in self.llm.chat_stream(state.messages):
                    delta = chunk.choices[0].delta.content
                    full_content += delta
                    yield chunk

                state.set_final_response(full_content)
                state.set_status(AgentStatus.FINISHED)
                break

        return _stream()
