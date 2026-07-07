"""FastAPI 应用入口。

使用新架构：
- AgentLoop（核心循环）
- LLMPlanner（决策层）
- ToolExecutor（执行层）
- LocalToolProvider（工具提供者）
- SimpleContextProvider（上下文提供者）
- DefaultPromptBuilder（提示词构建）
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ai_agent.core.agent_loop import AgentLoop
from ai_agent.core.application_profile import ApplicationProfile
from ai_agent.core.context_provider import SimpleContextProvider
from ai_agent.core.conversation import Conversation, ConversationStore
from ai_agent.core.event import EventBus, get_default_bus
from ai_agent.core.executor import ToolExecutor
from ai_agent.core.knowledge.file_knowledge_provider import FileKnowledgeProvider
from ai_agent.core.planner import LLMPlanner
from ai_agent.core.stream import item_to_sse_line
from ai_agent.llm.base import BaseLLM
from ai_agent.llm.factory import create_llm
from ai_agent.tools.base import ToolRegistry
from ai_agent.tools.calculator import CalculatorTool
from ai_agent.tools.datetime_tool import DateTimeTool
from ai_agent.tools.local_provider import LocalToolProvider
from ai_agent.tools.text_stats_tool import TextStatsTool

# ---------------------------------------------------------------------------
# 路径 / 常量
# ---------------------------------------------------------------------------

_project_root = Path(__file__).resolve().parent.parent.parent
_frontend_html = _project_root / "frontend" / "index.html"
_persist_path = _project_root / "data" / "conversations.db"
_max_conversations = 100
_max_agent_iterations = 5

# ---------------------------------------------------------------------------
# 应用定义（Application Profiles）
#
# ⬇ 这里是"做新应用"的唯一位置：加一个 Profile，框架自动把它跑起来。
# ---------------------------------------------------------------------------


def _default_profiles() -> Dict[str, ApplicationProfile]:
    """默认注册一个 Agent 应用。

    做一个新应用 = 在这个函数里加一行：
        profiles["my_app"] = ApplicationProfile.agent_app(
            name="my_app",
            system_prompt="my_app",       # prompts/my_app.txt
            tools=["calculator", "datetime"],
        )
    """
    return {
        "agent": ApplicationProfile.agent_app(
            "agent",
            system_prompt="agent_system",
            tools=["calculator", "datetime", "text_stats"],
            max_iterations=_max_agent_iterations,
        ),
    }


# ---------------------------------------------------------------------------
# 依赖构建（显式、可测试）
# ---------------------------------------------------------------------------


def build_tool_registry() -> ToolRegistry:
    """集中注册所有工具。加新工具时只改这里。"""
    registry = ToolRegistry()

    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(TextStatsTool())

    return registry


def build_app_state(
    llm: Optional[BaseLLM] = None,
    bus: Optional[EventBus] = None,
    store: Optional[ConversationStore] = None,
    tool_registry: Optional[ToolRegistry] = None,
    profiles: Optional[Dict[str, ApplicationProfile]] = None,
) -> "AppState":
    """构建完整的运行时依赖图。

    流程：
      1. 创建基础基础设施（LLM / EventBus / KnowledgeProvider / ConversationStore / ToolRegistry）
      2. 读取 Application Profiles —— 每个 Profile 描述一个 Agent 应用
      3. 按 Profile 构造 AgentLoop

    做一个新应用 → 新增 Application Profile；不用改这里的任何一行。
    """
    bus = bus or get_default_bus()

    knowledge_provider = FileKnowledgeProvider(_project_root / "data" / "knowledge")
    import asyncio

    asyncio.run(knowledge_provider.setup())

    store = store or ConversationStore(
        max_conversations=_max_conversations,
        persist_path=_persist_path,
        knowledge_provider=knowledge_provider,
    )
    llm = llm or create_llm()
    tool_registry = tool_registry or build_tool_registry()

    profiles = profiles or _default_profiles()

    agent_loops: Dict[str, AgentLoop] = {}

    for profile in profiles.values():
        system_prompt_text = profile.resolve_system_prompt()

        # Agent 应用：构造 Planner + Executor + AgentLoop
        resolved_tools = profile.resolve_tools(tool_registry)
        local_registry = ToolRegistry()
        for t in resolved_tools:
            local_registry.register(t)
        tool_provider = LocalToolProvider(local_registry)
        tool_executor = ToolExecutor(tool_provider)
        context_provider = SimpleContextProvider(tool_provider, store)
        planner = LLMPlanner(
            llm=llm,
            tool_provider=tool_provider,
            system_prompt=system_prompt_text,
        )
        agent_loops[profile.name] = AgentLoop(
            planner=planner,
            executor=tool_executor,
            context_provider=context_provider,
            llm=llm,
            bus=bus,
        )

    return AppState(
        llm=llm,
        bus=bus,
        store=store,
        tool_registry=tool_registry,
        profiles=profiles,
        agent_loops=agent_loops,
    )


class AppState:
    """所有应用级单例的容器，挂在 app.state.ai 上。"""

    def __init__(
        self,
        llm: BaseLLM,
        bus: EventBus,
        store: ConversationStore,
        tool_registry: ToolRegistry,
        profiles: Dict[str, ApplicationProfile],
        agent_loops: Dict[str, AgentLoop],
    ) -> None:
        self.llm = llm
        self.bus = bus
        self.store = store
        self.tool_registry = tool_registry
        self.profiles = profiles
        self.agent_loops = agent_loops

    @property
    def agent_loop(self) -> AgentLoop:
        if "agent" in self.agent_loops:
            return self.agent_loops["agent"]
        if not self.agent_loops:
            raise RuntimeError(
                "没有注册任何 Agent 模式的应用。请在 _default_profiles() 中添加"
                "一个 ApplicationProfile.agent_app()。"
            )
        return next(iter(self.agent_loops.values()))


# ---------------------------------------------------------------------------
# lifespan：启动时创建依赖，关闭时清理
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    state = build_app_state()
    fastapi_app.state.ai = state
    yield


app = FastAPI(title="AI Agent API", version="0.2.0", lifespan=lifespan)

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
# Agent 聊天 API（唯一入口，永远流式）
# ---------------------------------------------------------------------------


@app.post("/agent/chat")
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


async def _agent_sse(state, session_id: str, user_input: str):
    """把 AgentLoop.run_stream 转成 SSE 文本流。"""
    async for sse_line in _to_sse(state.agent_loop.run_stream(session_id, user_input)):
        yield sse_line


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


async def _to_sse(stream):
    """把 StreamItem 流转成 SSE 文本流。"""
    async for item in stream:
        yield item_to_sse_line(item)


def _find_token_handler(bus: EventBus):
    """从 bus 里找 TokenCountHandler。"""
    for h in getattr(bus, "_handlers", []):
        from ai_agent.core.event import TokenCountHandler

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
