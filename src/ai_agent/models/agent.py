from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class AgentAction(BaseModel):
    action_type: Literal["thought", "tool_call", "finish", "error"]
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    is_final: bool = False


class AgentThought(BaseModel):
    thought: str
    next_action: str
    confidence: float = 0.0