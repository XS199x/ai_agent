from typing import List, Optional

from ai_agent.models.chat import ChatMessage
from ai_agent.models.context import AgentContext


def build_messages(
    context: AgentContext,
    system_prompt: str,
    include_tool_messages: bool = True,
) -> List[ChatMessage]:
    """Build message history from context.

    Args:
        context: Agent context containing conversation history.
        system_prompt: System prompt to use.
        include_tool_messages: If True, include tool messages with validation.

    Returns:
        List of chat messages ready for LLM.
    """
    system_content = system_prompt
    snippets = (getattr(context, "system_prompt_snippets", "") or "").strip()
    if snippets:
        system_content = f"{system_content.rstrip()}\n\n{snippets}"

    messages: List[ChatMessage] = [ChatMessage(role="system", content=system_content)]

    if include_tool_messages:
        last_has_tool_calls = False
        for msg in context.conversation:
            if msg.role == "tool":
                if last_has_tool_calls:
                    messages.append(msg)
                last_has_tool_calls = False
            else:
                messages.append(msg)
                last_has_tool_calls = bool(msg.tool_calls)
    else:
        for msg in context.conversation:
            if msg.role in ("user", "assistant") and msg.content:
                messages.append(ChatMessage(role=msg.role, content=msg.content))

    return messages
