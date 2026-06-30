from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.ai_agent.core.agent_runtime import AgentRuntime
from src.ai_agent.core.chat_runtime import ChatRuntime
from src.ai_agent.core.conversation import Conversation, ConversationStore
from src.ai_agent.core.event import EventBus, get_default_bus
from src.ai_agent.core.planner import Planner
from src.ai_agent.core.stream import item_to_sse_line
from src.ai_agent.llm.base import BaseLLM
from src.ai_agent.llm.factory import create_llm
from src.ai_agent.models.chat import ChatMessage
from src.ai_agent.tools.base import ToolRegistry
from src.ai_agent.tools.calculator import CalculatorTool

# ---------------------------------------------------------------------------
# 路径 / 常量
# ---------------------------------------------------------------------------

_project_root = Path(__file__).resolve().parent.parent.parent
_frontend_html = _project_root / "frontend" / "index.html"
_persist_path = _project_root / "data" / "conversations.db"
_max_conversations = 100
_max_agent_iterations = 5


# ---------------------------------------------------------------------------
# 依赖构建（显式、可测试）
# ---------------------------------------------------------------------------


def build_tool_registry() -> ToolRegistry:
    """集中注册所有工具。加新工具时只改这里。"""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def build_app_state(
    llm: Optional[BaseLLM] = None,
    bus: Optional[EventBus] = None,
    store: Optional[ConversationStore] = None,
    tool_registry: Optional[ToolRegistry] = None,
) -> "AppState":
    """构建完整的运行时依赖图。测试时可以传 mock。"""
    bus = bus or get_default_bus()
    store = store or ConversationStore(
        max_conversations=_max_conversations, persist_path=_persist_path
    )
    llm = llm or create_llm()
    tool_registry = tool_registry or build_tool_registry()
    chat_runtime = ChatRuntime(llm=llm, store=store, bus=bus)
    planner = Planner(llm=llm, registry=tool_registry)
    agent_runtime = AgentRuntime(
        chat_runtime=chat_runtime,
        planner=planner,
        tool_registry=tool_registry,
        max_iterations=_max_agent_iterations,
    )
    return AppState(
        llm=llm,
        bus=bus,
        store=store,
        tool_registry=tool_registry,
        chat_runtime=chat_runtime,
        planner=planner,
        agent_runtime=agent_runtime,
    )


class AppState:
    """所有应用级单例的容器，挂在 app.state 上。"""

    def __init__(
        self,
        llm: BaseLLM,
        bus: EventBus,
        store: ConversationStore,
        tool_registry: ToolRegistry,
        chat_runtime: ChatRuntime,
        planner: Planner,
        agent_runtime: AgentRuntime,
    ) -> None:
        self.llm = llm
        self.bus = bus
        self.store = store
        self.tool_registry = tool_registry
        self.chat_runtime = chat_runtime
        self.planner = planner
        self.agent_runtime = agent_runtime


# ---------------------------------------------------------------------------
# lifespan：启动时创建依赖，关闭时清理
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    state = build_app_state()
    # 挂到 FastAPI 原生 state，测试时可以 override
    fastapi_app.state.ai = state
    yield
    # 在这里加清理逻辑（比如关闭 DB、刷新日志文件等）


app = FastAPI(title="AI Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def state_of(request: Request) -> AppState:
    """在路由里从 request.app.state.ai 拿依赖。"""
    return request.app.state.ai


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    messages: list
    stream: bool = True
    session_id: Optional[str] = None


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    initial_messages: Optional[list] = None


class RenameConversationRequest(BaseModel):
    title: str


class UpdateSystemPromptRequest(BaseModel):
    system_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# 根路由 + 健康检查
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root():
    if _frontend_html.exists():
        return _frontend_html.read_text(encoding="utf-8")
    return HTMLResponse(
        content="""
        <html><body style='font-family:sans-serif;padding:40px;text-align:center'>
        <h2>🤖 AI Agent API</h2>
        <p>请在 <code>frontend/index.html</code> 创建前端页面。</p>
        <p><a href='/docs'>查看 API 文档 →</a></p>
        </body></html>
        """
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/tools")
async def list_tools(request: Request) -> dict:
    state = state_of(request)
    tools = [t.to_dict() for t in state.tool_registry.all()]
    return {"tools": tools, "count": len(tools)}


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------


@app.get("/conversations")
async def list_conversations(request: Request) -> dict:
    state = state_of(request)
    convs = state.store.list_all()
    return {"conversations": [_summary(c, state) for c in convs], "count": len(convs)}


@app.post("/conversations")
async def create_conversation(request: Request, req: CreateConversationRequest) -> dict:
    state = state_of(request)
    conv = state.store.create(title=req.title, system_prompt=req.system_prompt)
    if req.initial_messages:
        conv.extend(req.initial_messages)
    return {"conversation": conv.to_dict()}


@app.get("/conversations/{session_id}")
async def get_conversation(request: Request, session_id: str) -> dict:
    state = state_of(request)
    conv = state.store.get(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@app.patch("/conversations/{session_id}/title")
async def rename_conversation(
    request: Request, session_id: str, req: RenameConversationRequest
) -> dict:
    state = state_of(request)
    conv = state.store.rename(session_id, req.title or "未命名对话")
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@app.patch("/conversations/{session_id}/system_prompt")
async def update_conversation_system_prompt(
    request: Request, session_id: str, req: UpdateSystemPromptRequest
) -> dict:
    state = state_of(request)
    conv = state.store.update_system_prompt(session_id, req.system_prompt)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@app.delete("/conversations/{session_id}/messages")
async def clear_conversation_messages(request: Request, session_id: str) -> dict:
    state = state_of(request)
    conv = state.store.clear_messages(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@app.delete("/conversations/{session_id}")
async def delete_conversation(request: Request, session_id: str) -> dict:
    state = state_of(request)
    if not state.store.delete(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True, "session_id": session_id}


@app.get("/conversations/{session_id}/stats")
async def get_conversation_stats(request: Request, session_id: str) -> dict:
    state = state_of(request)
    conv = state.store.get(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    token_stats = _token_stats_for(session_id, state.bus)
    return {
        "session_id": session_id,
        "message_count": len(conv.messages),
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
    }


# ---------------------------------------------------------------------------
# 聊天 API
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat(request: Request, req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    state = state_of(request)
    user_messages = _coerce_messages(req.messages)

    if req.stream:
        return StreamingResponse(
            _to_sse(state.chat_runtime.chat_stream(req.session_id, user_messages)),
            media_type="text/event-stream",
        )

    text = await state.chat_runtime.chat(req.session_id, user_messages)
    return _build_completion_response(text)


@app.post("/agent/chat")
async def agent_chat(request: Request, req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    state = state_of(request)
    user_messages = _coerce_messages(req.messages)

    if req.stream:
        return StreamingResponse(
            _to_sse(state.agent_runtime.run_stream(req.session_id, user_messages)),
            media_type="text/event-stream",
        )

    text = await state.agent_runtime.run(req.session_id, user_messages)
    return _build_completion_response(text)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


async def _to_sse(stream):
    """把 StreamItem 流转成 SSE 文本流。"""
    async for item in stream:
        yield item_to_sse_line(item)


def _coerce_messages(messages: list) -> list:
    """把请求里的消息统一转成 ChatMessage（容错：dict/list/ChatMessage 都接受）。"""
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append(
                ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
            )
        elif isinstance(m, ChatMessage):
            out.append(m)
        else:
            out.append(ChatMessage(role="user", content=str(m)))
    return out


def _build_completion_response(text: str) -> dict:
    """非流式返回的响应结构（和 OpenAI SDK 风格一致）。"""
    return {
        "id": "non-stream",
        "object": "chat.completion",
        "created": int(time()),
        "model": "",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _find_token_handler(bus: EventBus):
    """从 bus 里找 TokenCountHandler（如果注册了的话）。"""
    for h in getattr(bus, "_handlers", []):
        # 延迟导入避免循环引用
        from src.ai_agent.core.event import TokenCountHandler

        if isinstance(h, TokenCountHandler):
            return h
    return None


def _token_stats_for(session_id: str, bus: EventBus) -> dict:
    handler = _find_token_handler(bus)
    if handler is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    stats = handler.get_latest(session_id) or {}
    pt = int(stats.get("prompt_tokens", 0) or 0)
    ct = int(stats.get("completion_tokens", 0) or 0)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def _summary(conv: "Conversation", state: "AppState") -> dict:
    """会话列表项的摘要。"""
    token_stats = _token_stats_for(conv.session_id, state.bus)
    d = conv.to_dict()
    d.update(
        {
            "message_count": len(conv.messages),
            "prompt_tokens": token_stats["prompt_tokens"],
            "completion_tokens": token_stats["completion_tokens"],
            "total_tokens": token_stats["total_tokens"],
        }
    )
    return d
