from typing import Optional

from pydantic import BaseModel


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
