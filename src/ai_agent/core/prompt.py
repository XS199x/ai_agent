import math
from typing import List, Optional, Union

from src.ai_agent.models.chat import ChatMessage


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：1 token ≈ 4 英文字符 ≈ 1.5 中文字符。
    不依赖任何 tokenizer，足够用于 history 截断的容量判断。
    """
    if not text:
        return 0
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4) + math.ceil(non_ascii_count * 2 / 3))


class PromptBuilder:
    """统一把"发给 LLM 的消息"构建出来。

    设计目标：
    - 链式调用、可测、可扩展。未来加 RAG、Memory、Tool 只需在这个类里加方法，
    而不是去改业务代码。对外行为：build() 永远只产 List[ChatMessage]，跟当前
    LLM API 完全兼容。

    新特性（A1）：history 支持按轮数 + token 数双重截断，避免超长上下文。

    典型用法：
        builder = PromptBuilder()
        builder.system("你是助手")
        builder.history([ChatMessage])
        builder.user("你好")
        messages = builder.build()
    """

    DEFAULT_MAX_TURNS: int = 20
    DEFAULT_MAX_TOKENS: int = 6000

    def __init__(
        self,
        max_history_turns: Optional[int] = None,
        max_history_tokens: Optional[int] = None,
    ) -> None:
        self._system: Optional[str] = None
        self._history: List[ChatMessage] = []
        self._user_messages: List[ChatMessage] = []
        # 截断参数：None 表示不启用该维度限制
        self._max_history_turns: Optional[int] = max_history_turns
        self._max_history_tokens: Optional[int] = max_history_tokens
        # 未来扩展：RAG、Memory（现在只做接口，暂不拼入 build）
        self._rag_contexts: List[str] = []
        self._memory_items: List[str] = []

    # ------------------------------------------------------------------
    # 基础方法
    # ------------------------------------------------------------------
    def system(self, prompt: Optional[str]) -> "PromptBuilder":
        """设置 system prompt。传空字符串 / None 表示不设置。"""
        if prompt:
            self._system = prompt
        return self

    def history(self, messages: List[ChatMessage]) -> "PromptBuilder":
        """设置历史消息（user/assistant/tool 等）。"""
        self._history = list(messages)
        return self

    def user(
        self, message: Union[str, ChatMessage, List[Union[str, ChatMessage]]]
    ) -> "PromptBuilder":
        """追加一条或一批 user 消息。

        - 字符串 → 自动包装为 ChatMessage(role="user", content=...)
        - ChatMessage → 直接使用（保留原 role，便于非 user 消息拼入）
        - list → 逐个处理
        """
        if isinstance(message, list):
            for m in message:
                self.user(m)
            return self
        if isinstance(message, str):
            self._user_messages.append(ChatMessage(role="user", content=message))
        else:
            self._user_messages.append(message)
        return self

    # ------------------------------------------------------------------
    # 扩展方法（下一版本才真正启用）
    # 接口先开放，方便将来逐步启用。
    # ------------------------------------------------------------------
    def rag_context(self, text: str) -> "PromptBuilder":
        if text:
            self._rag_contexts.append(text)
        return self

    def memory_item(self, text: str) -> "PromptBuilder":
        if text:
            self._memory_items.append(text)
        return self

    # ------------------------------------------------------------------
    # 截断逻辑（A1）
    # ------------------------------------------------------------------
    def _truncate_history(
        self,
        history: List[ChatMessage],
        max_turns: Optional[int],
        max_tokens: Optional[int],
    ) -> List[ChatMessage]:
        """从最新消息往前截断：
        - max_turns：最多保留 N 轮（每 2 条消息算 1 轮，这里直接按消息条数的 2 倍处理）
        - max_tokens：累计 token 数不超过该值
        - system prompt 和 user_messages 不计入 history，但 system prompt 占用的 token 会
          从 max_tokens 中预留
        """
        if not history:
            return []

        system_tokens = _estimate_tokens(self._system or "")
        user_tokens = sum(_estimate_tokens(m.content) for m in self._user_messages)
        budget = None
        if max_tokens is not None:
            budget = max(100, max_tokens - system_tokens - user_tokens)

        # 按条数（轮数）初步筛选：保留最后 N*2 条消息
        turns = max_turns if max_turns is not None else self.DEFAULT_MAX_TURNS
        candidate = list(history[-turns * 2 :]) if turns > 0 else list(history)

        # 按 token 预算再剪：从最新一条开始往回累加
        if budget is not None:
            kept: List[ChatMessage] = []
            used = 0
            for msg in reversed(candidate):
                tok = _estimate_tokens(msg.content)
                if used + tok > budget and kept:
                    # 超出预算且已经至少保留了一条 → 停
                    break
                kept.append(msg)
                used += tok
            candidate = list(reversed(kept))

        return candidate

    # ------------------------------------------------------------------
    # 产出
    # ------------------------------------------------------------------
    def build(self) -> List[ChatMessage]:
        out: List[ChatMessage] = []

        if self._system:
            out.append(ChatMessage(role="system", content=self._system))

        # 当前版本：RAG / Memory 暂不注入（下一版本启用）
        # 这里留好接口，调用方调用 builder.rag_context() 也不会坏

        if self._history:
            truncated = self._truncate_history(
                self._history, self._max_history_turns, self._max_history_tokens
            )
            out.extend(truncated)

        if self._user_messages:
            out.extend(self._user_messages)

        return out

    # ------------------------------------------------------------------
    # 调试辅助
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"PromptBuilder(system={bool(self._system)}, "
            f"history={len(self._history)}, "
            f"user_messages={len(self._user_messages)}, "
            f"rag={len(self._rag_contexts)}, "
            f"memory={len(self._memory_items)})"
        )
