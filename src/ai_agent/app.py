"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ai_agent.dependencies import AppState, build_app_state
from ai_agent.routes import agent_router, conversation_router

_project_root = Path(__file__).resolve().parent.parent.parent
_frontend_html = _project_root / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    state = build_app_state()
    await state.setup()
    fastapi_app.state.ai = state
    try:
        yield
    finally:
        await state.teardown()


app = FastAPI(title="AI Agent API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation_router)
app.include_router(agent_router)


def state_of(request: Request) -> AppState:
    return request.app.state.ai


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
