"""AgentRuntime：Planner → Tool → 回答 的决策循环。

架构位置：
    Browser → FastAPI → AgentRuntime → ChatRuntime → ...

执行流程（每次请求）：
1. Planner 看完整上下文（用户消息 + 已追加的工具结果）→ 判断是否调用工具
2. 如果要调用工具：执行工具，把工具结果用 role=user + XML 标记追加
3. 不需要工具时：调 ChatRuntime 生成最终回答

消息格式设计原则（避免 role=tool 的 tool_call_id 问题）：
- 工具结果 = role=user + XML 标记 `<tool_result tool="...">...</tool_result>`
- 工具意图 = role=assistant + 简短自然语言声明
- 这样 LLM 不需要理解原生 function calling 协议
"""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator, List, Optional

from src.ai_agent.core.chat_runtime import ChatRuntime
from src.ai_agent.core.event import Event, EventBus
from src.ai_agent.core.planner import Planner, PlannerDecision
from src.ai_agent.core.stream import StreamItem
from src.ai_agent.models.chat import ChatMessage
from src.ai_agent.tools.base import ToolRegistry


class AgentRuntime:
    def __init__(
        self,
        chat_runtime: ChatRuntime,
        planner: Planner,
        tool_registry: Optional[ToolRegistry] = None,
        bus: Optional[EventBus] = None,
        max_iterations: int = 5,
    ) -> None:
        self.chat = chat_runtime
        self.planner = planner
        self.registry = tool_registry or ToolRegistry()
        self.bus = bus
        self.max_iterations = max_iterations

    # ---------- 内部：发事件（安全：没有 bus 也不炸） ----------

    def _emit(self, name: str, payload: dict) -> None:
        if self.bus is not None:
            self.bus.emit(Event(name=name, payload=payload))

    # ---------- 公共：非流式 ----------

    async def run(
        self,
        session_id: Optional[str],
        user_messages: List[ChatMessage],
    ) -> str:
        """走完整 Agent 流程。工具调用过程会走 EventBus（如果传了 bus）。"""
        extra_messages: List[ChatMessage] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # 步骤 A：Planner 决策
            self._emit(
                "agent.planning",
                {
                    "iteration": iteration,
                    "session_id": session_id,
                    "available_tools": [t.name for t in self.registry.all()],
                },
            )

            decision: PlannerDecision = await self.planner.plan(
                list(user_messages) + extra_messages
            )

            # 步骤 B：不需要工具 → 结束决策循环
            if not decision.use_tool:
                self._emit(
                    "agent.decision",
                    {
                        "session_id": session_id,
                        "use_tool": False,
                        "reason": decision.reason,
                    },
                )
                break

            tool = self.registry.get(decision.tool or "")
            if tool is None:
                self._emit(
                    "agent.error",
                    {
                        "session_id": session_id,
                        "message": f"Planner 选择了不存在的工具：{decision.tool!r}",
                    },
                )
                break

            # 步骤 C：调用工具
            self._emit(
                "agent.tool_call",
                {
                    "session_id": session_id,
                    "tool": decision.tool,
                    "args": decision.args,
                    "reason": decision.reason,
                },
            )

            try:
                result = tool.run(decision.args or {})
            except Exception as e:
                result = f"调用工具时出错：{type(e).__name__}: {e}"
                self._emit(
                    "agent.tool_error",
                    {"session_id": session_id, "message": str(result)},
                )

            self._emit(
                "agent.tool_result",
                {
                    "session_id": session_id,
                    "tool": decision.tool,
                    "result": str(result),
                },
            )

            # 工具意图声明（assistant，让对话更自然）
            extra_messages.append(
                ChatMessage(
                    role="assistant",
                    content=(
                        f"好的，我来调用 {decision.tool} 工具。"
                        f"（参数：{json.dumps(decision.args, ensure_ascii=False)}）"
                    ),
                )
            )

            # 工具结果（role=user + XML 标记，让 LLM 明确知道这是外部输入）
            extra_messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        f'<tool_result tool="{decision.tool}">{result}</tool_result>'
                    ),
                )
            )

        # 生成最终回答
        final_messages = list(user_messages) + extra_messages
        return await self.chat.chat(session_id, final_messages)

    # ---------- 公共：流式 ----------

    async def run_stream(
        self,
        session_id: Optional[str],
        user_messages: List[ChatMessage],
    ) -> AsyncGenerator[StreamItem, None]:
        """流式版本。Planner/工具调用阶段以 event 形式推给前端，同时发到 EventBus。"""
        extra_messages: List[ChatMessage] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # 步骤 A：Planner 决策（yield + emit）
            event_payload = {
                "iteration": iteration,
                "session_id": session_id,
                "available_tools": [t.name for t in self.registry.all()],
            }
            self._emit("agent.planning", event_payload)
            yield StreamItem(
                kind="event",
                event_name="agent.planning",
                event_payload=event_payload,
                created_at=time.time(),
            )

            decision: PlannerDecision = await self.planner.plan(
                list(user_messages) + extra_messages
            )

            self._emit("agent.planning.decision", decision.to_json())

            # 步骤 B：不需要工具 → 结束决策循环
            if not decision.use_tool:
                decision_payload = {
                    "session_id": session_id,
                    "use_tool": False,
                    "reason": decision.reason,
                }
                self._emit("agent.decision", decision_payload)
                yield StreamItem(
                    kind="event",
                    event_name="agent.decision",
                    event_payload=decision_payload,
                    created_at=time.time(),
                )
                break

            tool = self.registry.get(decision.tool or "")
            if tool is None:
                err_payload = {
                    "session_id": session_id,
                    "message": f"Planner 选择了不存在的工具：{decision.tool!r}",
                }
                self._emit("agent.error", err_payload)
                yield StreamItem(
                    kind="event",
                    event_name="agent.error",
                    event_payload=err_payload,
                    created_at=time.time(),
                )
                break

            # 步骤 C：调用工具（yield + emit）
            tool_call_payload = {
                "session_id": session_id,
                "tool": decision.tool,
                "args": decision.args,
                "reason": decision.reason,
            }
            self._emit("agent.tool_call", tool_call_payload)
            yield StreamItem(
                kind="event",
                event_name="agent.tool_call",
                event_payload=tool_call_payload,
                created_at=time.time(),
            )

            try:
                result = tool.run(decision.args or {})
            except Exception as e:
                result = f"调用工具时出错：{type(e).__name__}: {e}"
                err_payload = {"session_id": session_id, "message": str(result)}
                self._emit("agent.tool_error", err_payload)

            tool_result_payload = {
                "session_id": session_id,
                "tool": decision.tool,
                "result": str(result),
            }
            self._emit("agent.tool_result", tool_result_payload)
            yield StreamItem(
                kind="event",
                event_name="agent.tool_result",
                event_payload=tool_result_payload,
                created_at=time.time(),
            )

            # 声明工具意图 → 追加到消息序列
            extra_messages.append(
                ChatMessage(
                    role="assistant",
                    content=(
                        f"好的，我来调用 {decision.tool} 工具。"
                        f"（参数：{json.dumps(decision.args, ensure_ascii=False)}）"
                    ),
                )
            )

            # 工具结果 → 追加到消息序列
            extra_messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        f'<tool_result tool="{decision.tool}">{result}</tool_result>'
                    ),
                )
            )

        # 生成最终回答（流式）
        final_messages = list(user_messages) + extra_messages
        async for item in self.chat.chat_stream(session_id, final_messages):
            yield item
