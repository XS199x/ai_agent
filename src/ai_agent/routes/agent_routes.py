from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ai_agent.core.stream import item_to_sse_line
from ai_agent.dependencies import AppState
from ai_agent.schemas import ChatRequest

router = APIRouter(prefix="/agent", tags=["agent"])


def state_of(request: Request) -> AppState:
    return request.app.state.ai


@router.post("/chat")
async def agent_chat(request: Request, req: ChatRequest):
    """Agent 对话。永远返回 SSE（text/event-stream）。"""
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    state = state_of(request)

    last_user_message = None
    for m in req.messages:
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_message = m.get("content", "")
        else:
            last_user_message = str(m)

    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    session = req.session_id or ""

    return StreamingResponse(
        _agent_sse(state, session, last_user_message),
        media_type="text/event-stream",
    )


async def _agent_sse(state: AppState, session_id: str, user_input: str):
    """把 AgentRuntime.run_stream 转成 SSE 文本流。"""
    async for sse_line in _to_sse(
        state.agent_runtime.run_stream(session_id, user_input)
    ):
        yield sse_line


async def _to_sse(stream):
    """把 StreamItem 流转成 SSE 文本流。"""
    async for item in stream:
        yield item_to_sse_line(item)
