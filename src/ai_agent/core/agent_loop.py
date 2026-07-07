"""AgentLoop：整个 Agent Runtime 的唯一核心。

使用 StreamHandle 作为流式输出的中间层：
- AgentLoop 调用 StreamHandle.emit_* 方法
- StreamHandle 内部写入 asyncio.Queue，同时 emit 到 EventBus
- FastAPI 消费 StreamHandle.stream() 转成 SSE

所有关键状态都通过 StreamHandle.emit_event 抛出去：
- log: 运行日志
- iteration: 迭代次数
- planner_result: Planner 决策结果
- tool_call: 工具调用
- tool_result: 工具结果
- tool_error: 工具错误
- llm_start: 开始生成回答
- llm_messages: 传给 LLM 的消息摘要
- llm_completed: 回答生成完成
- done: 结束
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

from ai_agent.core.executor import ActionExecutor, ContextProvider
from ai_agent.core.planner import Planner
from ai_agent.core.stream import StreamHandle, StreamItem
from ai_agent.llm.base import BaseLLM
from ai_agent.models.action import Action, AnswerAction, ErrorAction, ToolAction
from ai_agent.models.chat import ChatMessage


class AgentLoop:
    """Agent 的核心循环。"""

    def __init__(
        self,
        planner: Planner,
        executor: ActionExecutor,
        context_provider: ContextProvider,
        llm: Optional[BaseLLM] = None,
        bus: Optional[Any] = None,
    ) -> None:
        from ai_agent.prompts.prompt_loader import load_prompt

        self._planner = planner
        self._executor = executor
        self._context_provider = context_provider
        self._llm = llm
        self._bus = bus
        self._answer_prompt = load_prompt(
            "agent/answer",
            default="你是一个智能助手。",
        )

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    async def run_stream(
        self, session_id: str, user_input: str
    ) -> AsyncGenerator[StreamItem, None]:
        """流式执行：返回 AsyncGenerator[StreamItem, None]。"""
        handle = StreamHandle(session_id=session_id, bus=self._bus)

        import asyncio

        asyncio.create_task(self._execute(session_id, user_input, handle))

        async for item in handle.stream():
            yield item

    # ------------------------------------------------------------------
    # 内部执行逻辑
    # ------------------------------------------------------------------

    async def _execute(
        self,
        session_id: str,
        user_input: str,
        handle: StreamHandle,
    ) -> None:
        """核心执行逻辑：决策 → 工具 → 回答。"""
        iteration = 0
        try:
            context = await self._context_provider.get_context(session_id, user_input)
            max_iter = context.runtime_state.max_iterations
            handle.emit_event(
                "log",
                {
                    "message": f"启动 session={session_id}, max_iterations={max_iter}",
                    "level": "info",
                },
            )

            while iteration < max_iter:
                iteration += 1

                # 1. 迭代开始
                handle.emit_event(
                    "iteration",
                    {"iteration": iteration, "message": f"第 {iteration} 步推理中..."},
                )

                # 2. Planner 决策
                action = await self._planner.plan(context)
                action_type = type(action).__name__
                action_info = {"type": action_type, "thought": action.thought}
                if isinstance(action, ToolAction):
                    action_info["tool"] = action.name
                    action_info["args"] = action.args
                elif isinstance(action, AnswerAction):
                    action_info["content_preview"] = (
                        action.content[:50] if action.content else ""
                    )
                elif isinstance(action, ErrorAction):
                    action_info["error"] = action.message

                handle.emit_event("planner_result", action_info)

                # ---- 分支 1：终止性动作 ----
                if action.is_terminal():
                    if isinstance(action, ErrorAction):
                        handle.emit_event(
                            "log",
                            {"message": f"错误: {action.message}", "level": "error"},
                        )
                        handle._error = action.message
                        handle.emit_token(f"错误：{action.message}")
                        handle.emit_done(
                            {
                                "success": False,
                                "iteration": iteration,
                                "error": action.message,
                            }
                        )
                        return

                    # AnswerAction: 直接回答
                    handle.emit_event(
                        "llm_start",
                        {"message": "开始生成答案...", "source": "direct_answer"},
                    )
                    messages = self._build_llm_messages(context)
                    handle.emit_event(
                        "llm_messages",
                        {
                            "count": len(messages),
                            "summary": self._summarize_messages(messages),
                        },
                    )
                    await self._streaming_answer(handle, messages)
                    handle.emit_event("llm_completed", {"message": "回答生成完成"})
                    handle._success = True
                    handle.emit_done({"success": True, "iteration": iteration})
                    return

                # ---- 分支 2：工具调用 ----
                if not isinstance(action, ToolAction):
                    handle.emit_event(
                        "log",
                        {
                            "message": f"非 ToolAction 无法执行: {type(action).__name__}",
                            "level": "error",
                        },
                    )
                    handle._error = "非 ToolAction 无法执行"
                    handle.emit_token("执行失败：无法识别的动作类型")
                    handle.emit_done(
                        {
                            "success": False,
                            "iteration": iteration,
                            "error": "invalid_action_type",
                        }
                    )
                    return

                tool_name = action.name
                tool_args = action.args

                handle.emit_event(
                    "tool_call",
                    {"tool": tool_name, "args": tool_args, "iteration": iteration},
                )

                try:
                    observation = await self._executor.execute(action)
                    handle.emit_event(
                        "tool_result",
                        {
                            "tool": tool_name,
                            "observation": str(observation)[:200],
                            "observation_full": str(observation),
                        },
                    )
                    context = context.with_action_result(action, observation)

                    # 工具执行完 —— 生成最终回答
                    handle.emit_event(
                        "llm_start",
                        {"message": "根据工具结果生成答案...", "source": "after_tool"},
                    )
                    messages = self._build_llm_messages(context)
                    handle.emit_event(
                        "llm_messages",
                        {
                            "count": len(messages),
                            "summary": self._summarize_messages(messages),
                        },
                    )
                    await self._streaming_answer(handle, messages)
                    handle.emit_event("llm_completed", {"message": "回答生成完成"})
                    handle._success = True
                    handle.emit_done({"success": True, "iteration": iteration})
                    return

                except Exception as e:
                    handle.emit_event(
                        "tool_error", {"tool": tool_name, "error": str(e)}
                    )
                    handle._error = str(e)
                    handle.emit_token(f"执行失败：{str(e)}")
                    handle.emit_done(
                        {"success": False, "iteration": iteration, "error": str(e)}
                    )
                    return

            # 最大迭代次数
            handle._error = "max_iterations_reached"
            handle.emit_done(
                {
                    "success": False,
                    "iteration": iteration,
                    "error": "max_iterations_reached",
                }
            )

        except Exception as e:
            handle._error = str(e)
            handle.emit_event(
                "log", {"message": f"异常: {type(e).__name__}: {e}", "level": "error"}
            )
            handle.emit_token(f"Agent 执行异常：{str(e)}")
            handle.emit_done(
                {"success": False, "iteration": iteration, "error": str(e)}
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _build_llm_messages(self, context: Any) -> List[ChatMessage]:
        """构建传给 LLM 的完整消息列表。"""
        messages: List[ChatMessage] = []
        messages.append(ChatMessage(role="system", content=self._answer_prompt))
        conversation = getattr(context, "conversation", [])
        if conversation:
            messages.extend(conversation)
        return messages

    def _summarize_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """生成消息摘要。"""
        summary = []
        for i, msg in enumerate(messages):
            summary.append(
                {
                    "role": msg.role,
                    "content_preview": msg.content[:80] if msg.content else "",
                }
            )
            if i >= 5:
                summary.append(
                    {
                        "role": "...",
                        "content_preview": f"+{len(messages) - 5} 条更多消息",
                    }
                )
                break
        return summary

    async def _streaming_answer(
        self, handle: StreamHandle, messages: List[ChatMessage]
    ) -> None:
        """调用 llm.chat_stream 流式生成回答，写入 StreamHandle。"""
        if self._llm is None:
            fallback = "（LLM 未配置）"
            handle.emit_event(
                "log",
                {
                    "message": f"LLM 未配置，使用兜底回答: {fallback}",
                    "level": "warning",
                },
            )
            for ch in fallback:
                handle.emit_token(ch)
            return

        try:
            await handle.consume_llm_chunk_stream(self._llm.chat_stream(messages))
        except Exception as e:
            handle.emit_event(
                "log", {"message": f"chat_stream 失败: {e}", "level": "error"}
            )
            fallback = f"(生成失败: {e})"
            for ch in fallback:
                handle.emit_token(ch)
