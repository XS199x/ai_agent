from typing import List, Optional

from src.ai_agent.models.chat import ChatMessage


class Planner:
    def __init__(self) -> None:
        pass

    def plan(self, messages: List[ChatMessage]) -> str:
        return "Directly respond to the user's query"

    def should_call_tool(self, messages: List[ChatMessage]) -> bool:
        return False

    def select_tool(self, messages: List[ChatMessage]) -> Optional[str]:
        return None

    def generate_tool_args(self, messages: List[ChatMessage], tool_name: str) -> dict:
        return {}
