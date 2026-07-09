"""AgentRuntime：Agent 运行时。

职责：
1. 控制 Agent 生命周期
2. 管理推理循环（多步）
3. 处理异常和中断
4. 管理迭代次数和超时

设计原则：
- 不包含业务逻辑，只做控制流
- 与 Planner/ContextManager/ActionDispatcher 解耦
- 支持真正的多步推理：工具结果回灌到上下文后再次交给 Planner

数据流：
用户输入 → ContextManager.build_initial() → AgentContext
           ↓
         Planner.plan() → Action
           ↓
         ActionDispatcher.dispatch() → Result
           ↓
         ContextManager.update() → 新 AgentContext
           ↓
         循环直到 Planner 返回 AnswerAction 或达到 max_iterations
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Optional

from ai_agent.core.action_dispatcher import ActionDispatcher
from ai_agent.core.context_manager import ContextManager
from ai_agent.core.planner import Planner
from ai_agent.core.stream import StreamHandle, StreamItem
from ai_agent.llm.base import BaseLLM
from ai_agent.models.action import Action, AnswerAction, ErrorAction, ToolAction
from ai_agent.models.context import AgentContext


class AgentRuntime:
    """Agent 运行时。"""

    def __init__(
        self,
        planner: Planner,
        context_manager: ContextManager,
        dispatcher: ActionDispatcher,
        llm: Optional[BaseLLM] = None,
        bus: Optional[Any] = None,
    ) -> None:
        self._planner = planner
        self._context_manager = context_manager
        self._dispatcher = dispatcher
        self._llm = llm
        self._bus = bus

    async def run_stream(
        self, session_id: str, user_input: str
    ) -> AsyncGenerator[StreamItem, None]:
        """流式执行：返回 AsyncGenerator[StreamItem, None]。"""
        handle = StreamHandle(session_id=session_id, bus=self._bus)
        task = asyncio.create_task(self._execute(session_id, user_input, handle))

        try:
            async for item in handle.stream():
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if task.done() and task.exception() is not None and not handle._done:
                handle.emit_error(str(task.exception()))

    async def _execute(
        self,
        session_id: str,
        user_input: str,
        handle: StreamHandle,
    ) -> None:
        """核心执行逻辑：多步推理循环。"""
        iteration = 0
        try:
            context = await self._context_manager.build_initial(session_id, user_input)
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

                handle.emit_event(
                    "iteration",
                    {"iteration": iteration, "message": f"第 {iteration} 步推理中..."},
                )

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

                    if isinstance(action, AnswerAction):
                        await self._handle_answer_action(handle, context, action)
                        handle._success = True
                        handle.emit_done({"success": True, "iteration": iteration})
                    else:
                        handle._error = f"未知的终止动作类型: {type(action).__name__}"
                        handle.emit_token(f"错误：未知的终止动作类型")
                        handle.emit_done(
                            {
                                "success": False,
                                "iteration": iteration,
                                "error": handle._error,
                            }
                        )
                    return

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
                    observation = await self._dispatcher.dispatch(action)
                    handle.emit_event(
                        "tool_result",
                        {
                            "tool": tool_name,
                            "observation": str(observation)[:200],
                            "observation_full": str(observation),
                        },
                    )

                    context = await self._context_manager.update(
                        context, action, observation
                    )

                    handle.emit_event(
                        "log",
                        {
                            "message": f"工具 {tool_name} 执行完成，上下文已更新，继续推理",
                            "level": "info",
                        },
                    )

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

            handle._error = "max_iterations_reached"
            handle.emit_token("达到最大迭代次数，推理结束。")
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
        finally:
            if not handle._done:
                handle._error = handle._error or "execution_aborted"
                handle.emit_done(
                    {"success": False, "iteration": iteration, "error": handle._error}
                )

    async def _handle_answer_action(
        self, handle: StreamHandle, context: AgentContext, action: AnswerAction
    ) -> None:
        """处理回答动作：生成最终答案。"""
        handle.emit_event(
            "llm_start",
            {"message": "开始生成答案...", "source": "direct_answer"},
        )
        messages = self._context_manager.build_llm_messages(context)
        handle.emit_event(
            "llm_messages",
            {
                "count": len(messages),
                "summary": self._context_manager.summarize_messages(messages),
            },
        )
        await self._streaming_answer(handle, messages)
        handle.emit_event("llm_completed", {"message": "回答生成完成"})

    async def _streaming_answer(self, handle: StreamHandle, messages: Any) -> None:
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
