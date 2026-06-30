"""AgentRuntime：在 ChatRuntime 之上加 Planner → Tool → 回答 的决策循环。

架构位置：
    Browser → FastAPI → AgentRuntime → ChatRuntime → ...

执行流程（每次请求）：
1. Planner 分析用户输入 → 判断是否调用工具
2. 如果要调用工具：执行工具 → 把"工具调用 + 工具结果"作为额外上下文消息
3. 调 ChatRuntime（它负责拼完整 Prompt、存历史、流式输出）

设计要点：
- 流模式：把 Planner/Tool 阶段以事件形式 yield 出去；然后直接迭代 chat_stream 的 token
- 非流模式：正常走 Planner → Tool → ChatRuntime.chat
"""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator, List, Optional

from src.ai_agent.core.chat_runtime import ChatRuntime
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
        max_iterations: int = 5,
    ) -> None:
        self.chat = chat_runtime
        self.planner = planner
        self.registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations

    # ---------- 非流式 ----------

    async def run(
        self,
        session_id: Optional[str],
        user_messages: List[ChatMessage],
    ) -> str:
        """走 Planner → Tool → ChatRuntime 的完整流程。非流式。"""
        extra_messages: List[ChatMessage] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            decision = await self.planner.plan(list(user_messages) + extra_messages)

            if not decision.use_tool:
                break

            tool = self.registry.get(decision.tool or "")
            if tool is None:
                extra_messages.append(
                    ChatMessage(
                        role="assistant",
                        content=f"[Agent：尝试调用未知工具 {decision.tool!r}，跳过]",
                    )
                )
                break

            try:
                result = tool.run(decision.args or {})
            except Exception as e:
                result = f"调用工具时出错：{type(e).__name__}: {e}"

            extra_messages.append(
                ChatMessage(
                    role="assistant",
                    content=(
                        f"[Agent：调用 {decision.tool}，"
                        f"参数 {json.dumps(decision.args, ensure_ascii=False)}，"
                        f"原因：{decision.reason}]"
                    ),
                )
            )
            extra_messages.append(
                ChatMessage(
                    role="tool",
                    content=f"工具 {decision.tool} 返回：{result}",
                    name=decision.tool,
                )
            )

        final_messages = list(user_messages) + extra_messages
        return await self.chat.chat(session_id, final_messages)

    # ---------- 流式 ----------

    async def run_stream(
        self,
        session_id: Optional[str],
        user_messages: List[ChatMessage],
    ) -> AsyncGenerator[StreamItem, None]:
        """流式版本。Planner/工具调用以 event 形式 yield。"""
        extra_messages: List[ChatMessage] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Step 1: 正在规划
            yield StreamItem(
                kind="event",
                event_name="agent.planning",
                event_payload={
                    "iteration": iteration,
                    "available_tools": [t.name for t in self.registry.all()],
                },
                created_at=time.time(),
            )

            decision: PlannerDecision = await self.planner.plan(
                list(user_messages) + extra_messages
            )

            # Step 2: 不需要工具 → 结束循环
            if not decision.use_tool:
                yield StreamItem(
                    kind="event",
                    event_name="agent.decision",
                    event_payload={
                        "use_tool": False,
                        "reason": decision.reason,
                    },
                    created_at=time.time(),
                )
                break

            tool = self.registry.get(decision.tool or "")
            if tool is None:
                yield StreamItem(
                    kind="event",
                    event_name="agent.error",
                    event_payload={
                        "message": f"Planner 选择了不存在的工具：{decision.tool!r}"
                    },
                    created_at=time.time(),
                )
                break

            yield StreamItem(
                kind="event",
                event_name="agent.tool_call",
                event_payload={
                    "tool": decision.tool,
                    "args": decision.args,
                    "reason": decision.reason,
                },
                created_at=time.time(),
            )

            # Step 3: 执行工具
            try:
                result = tool.run(decision.args or {})
            except Exception as e:
                result = f"调用工具时出错：{type(e).__name__}: {e}"
                yield StreamItem(
                    kind="event",
                    event_name="agent.tool_error",
                    event_payload={"message": str(result)},
                    created_at=time.time(),
                )

            yield StreamItem(
                kind="event",
                event_name="agent.tool_result",
                event_payload={"tool": decision.tool, "result": str(result)},
                created_at=time.time(),
            )

            extra_messages.append(
                ChatMessage(
                    role="assistant",
                    content=(
                        f"[Agent：调用 {decision.tool}，"
                        f"参数 {json.dumps(decision.args, ensure_ascii=False)}，"
                        f"原因：{decision.reason}]"
                    ),
                )
            )
            extra_messages.append(
                ChatMessage(
                    role="tool",
                    content=f"工具 {decision.tool} 返回：{result}",
                    name=decision.tool,
                )
            )

        # Step 4: 调 ChatRuntime 生成最终回答（流式）
        final_messages = list(user_messages) + extra_messages
        async for item in self.chat.chat_stream(session_id, final_messages):
            yield item
