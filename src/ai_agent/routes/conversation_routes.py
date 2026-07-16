from fastapi import APIRouter, HTTPException, Request

from ai_agent.core.event import EventBus
from ai_agent.core.handlers import TokenCountHandler
from ai_agent.dependencies import AppState
from ai_agent.models.chat import ChatMessage
from ai_agent.persistence.models import Conversation
from ai_agent.schemas import (
    CreateConversationRequest,
    RenameConversationRequest,
    UpdateSystemPromptRequest,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


def state_of(request: Request) -> AppState:
    return request.app.state.ai


def _find_token_handler(bus: EventBus) -> TokenCountHandler | None:
    for h in getattr(bus, "_handlers", []):
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


def _summary(conv: Conversation, state: AppState) -> dict:
    token_stats = _token_stats_for(conv.session_id, state.bus)
    d = conv.to_dict()
    d.update({
        "message_count": len(conv.messages),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
    })
    return d


@router.get("")
async def list_conversations(request: Request) -> dict:
    state = state_of(request)
    convs = state.store.list_all()
    return {"conversations": [_summary(c, state) for c in convs], "count": len(convs)}


@router.post("")
async def create_conversation(request: Request, req: CreateConversationRequest) -> dict:
    state = state_of(request)
    conv = state.store.create(title=req.title, system_prompt=req.system_prompt)
    if req.initial_messages:
        messages = []
        for m in req.initial_messages:
            if isinstance(m, dict):
                messages.append(
                    ChatMessage(
                        role=m.get("role", "user"), content=m.get("content", "")
                    )
                )
            else:
                messages.append(ChatMessage(role="user", content=str(m)))
        conv.extend(messages)
    return {"conversation": conv.to_dict()}


@router.get("/{session_id}")
async def get_conversation(request: Request, session_id: str) -> dict:
    state = state_of(request)
    conv = state.store.get(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@router.patch("/{session_id}/title")
async def rename_conversation(
    request: Request, session_id: str, req: RenameConversationRequest
) -> dict:
    state = state_of(request)
    conv = state.store.rename(session_id, req.title or "未命名对话")
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@router.patch("/{session_id}/system_prompt")
async def update_conversation_system_prompt(
    request: Request, session_id: str, req: UpdateSystemPromptRequest
) -> dict:
    state = state_of(request)
    conv = state.store.update_system_prompt(session_id, req.system_prompt)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@router.delete("/{session_id}/messages")
async def clear_conversation_messages(request: Request, session_id: str) -> dict:
    state = state_of(request)
    conv = state.store.clear_messages(session_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conv.to_dict()}


@router.delete("/{session_id}")
async def delete_conversation(request: Request, session_id: str) -> dict:
    state = state_of(request)
    if not state.store.delete(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True, "session_id": session_id}


@router.get("/{session_id}/stats")
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
