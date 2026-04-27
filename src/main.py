import uuid
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.config import settings
from src.models.schemas import ChatRequest, ChatResponse, HealthResponse
from src.vectorstore.connection import init_pool, close_pool
from src.memory.store import init_redis, close_redis
from src.graph.workflow import run_workflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    await init_redis()
    yield
    await close_pool()
    await close_redis()


app = FastAPI(
    title="CloudOrder Customer Service",
    description="Multi-agent customer service system for CloudOrder",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    response_text, intent = await run_workflow(
        message=request.message,
        session_id=session_id,
        user_id=request.user_id,
    )
    return ChatResponse(
        response=response_text,
        session_id=session_id,
        intent=intent,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        response_text, intent = await run_workflow(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id,
        )
        import json

        yield {
            "event": "message",
            "data": json.dumps(
                {
                    "response": response_text,
                    "session_id": session_id,
                    "intent": intent,
                }
            ),
        }
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())
