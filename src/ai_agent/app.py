import json
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.ai_agent.core.agent import Agent
from src.ai_agent.core.conversation import Conversation, ConversationStore
from src.ai_agent.llm.factory import create_llm
from src.ai_agent.models.chat import ChatMessage

app = FastAPI(title="AI Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_project_root = Path(__file__).resolve().parent.parent.parent
_frontend_html = _project_root / "frontend" / "index.html"
_persist_file = _project_root / "data" / "conversations.db"


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


_llm = None
_agent = None
_store = ConversationStore(max_conversations=100, persist_path=_persist_file)


def _get_llm():
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


def _get_agent():
    global _agent
    if _agent is None:
        _agent = Agent(llm=_get_llm())
    return _agent


def _get_store() -> ConversationStore:
    return _store


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True
    session_id: Optional[str] = None


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    initial_messages: Optional[List[ChatMessage]] = None


class RenameConversationRequest(BaseModel):
    title: str


class UpdateSystemPromptRequest(BaseModel):
    system_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# 会话管理 API
# ---------------------------------------------------------------------------


@app.get("/conversations")
async def list_conversations() -> dict:
    store = _get_store()
    convs = store.list_all()
    return {
        "conversations": [_summary(c) for c in convs],
        "count": len(convs),
    }


@app.post("/conversations")
async def create_conversation(req: CreateConversationRequest) -> dict:
    store = _get_store()
    conv = store.create(title=req.title, system_prompt=req.system_prompt)
    if req.initial_messages:
        conv.extend(req.initial_messages)
    return {"conversation": conv.to_dict()}


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str) -> dict:
    store = _get_store()
    conv = store.get(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@app.patch("/conversations/{session_id}/title")
async def rename_conversation(session_id: str, req: RenameConversationRequest) -> dict:
    store = _get_store()
    conv = store.rename(session_id, req.title or "未命名对话")
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@app.patch("/conversations/{session_id}/system_prompt")
async def update_conversation_system_prompt(
    session_id: str, req: UpdateSystemPromptRequest
) -> dict:
    store = _get_store()
    conv = store.update_system_prompt(session_id, req.system_prompt)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@app.delete("/conversations/{session_id}/messages")
async def clear_conversation_messages(session_id: str) -> dict:
    store = _get_store()
    conv = store.clear_messages(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@app.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str) -> dict:
    store = _get_store()
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# 聊天 API（支持 session_id 自动拼接历史）
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    effective_messages, conv = _bind_history(request.session_id, request.messages)
    llm = _get_llm()

    if request.stream:
        return StreamingResponse(
            _chat_stream_and_save(request.session_id, request.messages, conv),
            media_type="text/event-stream",
        )

    response = await llm.chat(effective_messages)
    if response.choices:
        _append_assistant_to_store(
            request.session_id, request.messages, response.choices[0].message
        )
    return response


@app.post("/agent/chat")
async def agent_chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    effective_messages, _ = _bind_history(request.session_id, request.messages)
    agent = _get_agent()

    if request.stream:
        return StreamingResponse(
            _agent_stream_and_save(request.session_id, request.messages),
            media_type="text/event-stream",
        )

    response = await agent.run(effective_messages)
    if response.choices:
        _append_assistant_to_store(
            request.session_id, request.messages, response.choices[0].message
        )
    return response


async def _chat_stream_and_save(
    session_id: Optional[str],
    request_messages: List[ChatMessage],
    conv: Optional[Conversation],
) -> AsyncGenerator[str, None]:
    llm = _get_llm()
    effective_messages, _ = _bind_history(session_id, request_messages)

    full_text = ""
    async for chunk in llm.chat_stream(effective_messages):
        delta = _chunk_delta_text(chunk)
        if delta:
            full_text += delta
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    if full_text:
        assistant_msg = ChatMessage(role="assistant", content=full_text)
        _append_assistant_to_store(session_id, request_messages, assistant_msg)


async def _agent_stream_and_save(
    session_id: Optional[str],
    request_messages: List[ChatMessage],
) -> AsyncGenerator[str, None]:
    agent = _get_agent()
    effective_messages, _ = _bind_history(session_id, request_messages)

    full_text = ""
    async for chunk in agent.run_stream(messages=effective_messages):
        delta = _chunk_delta_text(chunk)
        if delta:
            full_text += delta
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    if full_text:
        assistant_msg = ChatMessage(role="assistant", content=full_text)
        _append_assistant_to_store(session_id, request_messages, assistant_msg)


def _chunk_delta_text(chunk) -> str:
    try:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return ""
        return getattr(delta, "content", "") or ""
    except Exception:
        return ""


def _bind_history(
    session_id: Optional[str],
    request_messages: List[ChatMessage],
) -> (List[ChatMessage], Optional[Conversation]):
    """根据 session_id 构造实际发送给模型的 messages 列表。

    - 若提供 session_id：从会话历史中取 system_prompt + 历史 user/assistant，再叠加本次请求的消息；
    - 若未提供：仅使用请求本身的 messages（保持向后兼容）。
    """
    if not session_id:
        return list(request_messages), None

    store = _get_store()
    conv = store.get(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    effective: List[ChatMessage] = []

    if conv.system_prompt:
        effective.append(ChatMessage(role="system", content=conv.system_prompt))

    # 历史中的 user/assistant/tool 参与上下文，system/developer 也保留
    for m in conv.messages:
        effective.append(m)

    effective.extend(request_messages)

    if not effective:
        raise HTTPException(status_code=400, detail="No messages to send")

    return effective, conv


def _append_assistant_to_store(
    session_id: Optional[str],
    request_messages: List[ChatMessage],
    assistant_message: ChatMessage,
) -> None:
    if not session_id:
        return
    store = _get_store()
    # 通过 store 统一调用以触发持久化
    if request_messages:
        store.extend_messages(session_id, list(request_messages))
    store.append_message(session_id, assistant_message)


def _summary(conv: Conversation) -> dict:
    first_user = next(
        (m.content for m in conv.messages if m.role == "user"),
        None,
    )
    return {
        "session_id": conv.session_id,
        "title": conv.title,
        "system_prompt": conv.system_prompt,
        "message_count": len(conv.messages),
        "preview": (first_user or "")[:60],
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
