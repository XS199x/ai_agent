"""ChatRuntime：主线 —— 完整的聊天运行时。

架构位置：
    Browser → FastAPI → ChatRuntime → (PromptBuilder / ConversationStore / EventBus / StreamHandle / BaseLLM)

流程：
    收到消息 → 读历史 → 拼 Prompt → 调 LLM → 存历史 → 返回（流式 or 非流式）

新特性（A1）：history 默认按 20 轮 / 6000 tokens 截断，避免超长上下文。
"""

import asyncio
from typing import AsyncGenerator, List, Optional

from src.ai_agent.core.conversation import ConversationStore
from src.ai_agent.core.event import Event, EventBus
from src.ai_agent.core.prompt import PromptBuilder, _estimate_tokens
from src.ai_agent.core.stream import StreamHandle, StreamItem
from src.ai_agent.llm.base import BaseLLM
from src.ai_agent.models.chat import ChatMessage


class ChatRuntime:
    """最朴素的聊天运行时。

    只做一件事：「给用户输入 → 返回 LLM 回复」，
    同时自动管理会话历史、Prompt 构建、流式输出、事件发布。
    """

    DEFAULT_MAX_TURNS: int = 20
    DEFAULT_MAX_TOKENS: int = 6000

    def __init__(
        self,
        llm: BaseLLM,
        store: ConversationStore,
        bus: Optional[EventBus] = None,
        max_history_turns: Optional[int] = None,
        max_history_tokens: Optional[int] = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.bus = bus
        self._max_turns = max_history_turns
        self._max_tokens = max_history_tokens

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        session_id: Optional[str],
        user_messages: List[ChatMessage],
    ) -> AsyncGenerator[StreamItem, None]:
        """流式聊天。

        返回 AsyncGenerator[StreamItem]，由 FastAPI 侧负责转成 SSE。
        完成后自动把用户消息和助手回复写入 store。
        """

        async def _generator() -> AsyncGenerator[StreamItem, None]:
            effective_messages = self._build_messages(session_id, user_messages)

            handle = StreamHandle(bus=self.bus, session_id=session_id)
            producer = asyncio.create_task(
                handle.consume_llm_chunk_stream(
                    self.llm.chat_stream(effective_messages)
                )
            )

            try:
                async for item in handle.stream():
                    yield item
            finally:
                if not producer.done():
                    producer.cancel()

            # 写回历史
            if handle.full_text and session_id:
                assistant_msg = ChatMessage(role="assistant", content=handle.full_text)
                self._append_to_store(session_id, user_messages, assistant_msg)

        return _generator()

    async def chat(
        self,
        session_id: Optional[str],
        user_messages: List[ChatMessage],
    ) -> str:
        """非流式聊天。内部仍然走 StreamHandle（确保同一套 producer 语义）。"""
        effective_messages = self._build_messages(session_id, user_messages)

        handle = StreamHandle(bus=self.bus, session_id=session_id)
        producer = asyncio.create_task(
            handle.consume_llm_chunk_stream(self.llm.chat_stream(effective_messages))
        )

        try:
            async for _ in handle.stream():
                pass
        finally:
            if not producer.done():
                producer.cancel()

        if handle.full_text and session_id:
            assistant_msg = ChatMessage(role="assistant", content=handle.full_text)
            self._append_to_store(session_id, user_messages, assistant_msg)

        return handle.full_text

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        session_id: Optional[str],
        user_messages: List[ChatMessage],
    ) -> List[ChatMessage]:
        """从 store 读 system prompt + 历史，和用户消息一起拼出发给 LLM 的 messages。"""
        builder = PromptBuilder(
            max_history_turns=self._max_turns or self.DEFAULT_MAX_TURNS,
            max_history_tokens=self._max_tokens or self.DEFAULT_MAX_TOKENS,
        )

        if session_id:
            conv = self.store.get(session_id)
            if conv is not None:
                if conv.system_prompt:
                    builder.system(conv.system_prompt)
                if conv.messages:
                    builder.history(list(conv.messages))

        builder.user(list(user_messages))
        messages = builder.build()

        # 事件：让外部关心 prompt 大小的 handler 能拿到统计
        if self.bus is not None:
            prompt_tokens = sum(_estimate_tokens(m.content) for m in messages)
            self.bus.emit(
                Event(
                    "chat.prompt_stats",
                    {
                        "session_id": session_id,
                        "prompt_messages": len(messages),
                        "prompt_tokens": prompt_tokens,
                    },
                )
            )

        return messages

    def _append_to_store(
        self,
        session_id: str,
        user_messages: List[ChatMessage],
        assistant_message: ChatMessage,
    ) -> None:
        """把本轮的用户消息和助手回复追加到会话。"""
        if user_messages:
            self.store.extend_messages(session_id, list(user_messages))
        self.store.append_message(session_id, assistant_message)
