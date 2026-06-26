import json
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.ai_agent.config import config
from src.ai_agent.core.agent import Agent
from src.ai_agent.llm.factory import create_llm

app = FastAPI(title="AI Agent API", version="0.1.0")

llm = create_llm()
agent = Agent(llm=llm)

from src.ai_agent.models.chat import ChatMessage


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatCompletionResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    if request.stream:
        return StreamingResponse(
            chat_stream_generator(request.messages),
            media_type="text/event-stream",
        )

    response = await llm.chat(request.messages)
    return response


from src.ai_agent.models.chat import ChatCompletionChunk


async def chat_stream_generator(messages: List[ChatMessage]) -> ChatCompletionChunk:
    async for chunk in llm.chat_stream(messages):
        yield chunk


from src.ai_agent.models.chat import ChatMessage


@app.post("/agent/chat")
async def agent_chat(request: ChatRequest) -> ChatCompletionResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    if request.stream:
        return StreamingResponse(
            agent_stream_generator(request.messages),
            media_type="text/event-stream",
        )

    response = await agent.run(messages=request.messages)
    return response


from src.ai_agent.models.chat import ChatCompletionResponse


async def agent_stream_generator(messages: List[ChatMessage]) -> str:
    async for chunk in agent.run_stream(messages=messages):
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
