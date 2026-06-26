from enum import Enum
from typing import List, Optional

from src.ai_agent.models.chat import ChatMessage


class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    GENERATING = "generating"
    FINISHED = "finished"


class AgentState:
    def __init__(self) -> None:
        self.messages: List[ChatMessage] = []
        self.status: AgentStatus = AgentStatus.IDLE
        self.thoughts: List[str] = []
        self.actions: List[dict] = []
        self.iteration: int = 0
        self.final_response: Optional[str] = None

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def add_thought(self, thought: str) -> None:
        self.thoughts.append(thought)

    def add_action(self, action: dict) -> None:
        self.actions.append(action)

    def increment_iteration(self) -> None:
        self.iteration += 1

    def set_status(self, status: AgentStatus) -> None:
        self.status = status

    def set_final_response(self, response: str) -> None:
        self.final_response = response

    def reset(self) -> None:
        self.messages = []
        self.status = AgentStatus.IDLE
        self.thoughts = []
        self.actions = []
        self.iteration = 0
        self.final_response = None
