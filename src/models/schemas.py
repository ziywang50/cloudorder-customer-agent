from pydantic import BaseModel, Field
import uuid


class ChatRequest(BaseModel):
    message: str
    user_id: str | None = None
    session_id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str


class HealthResponse(BaseModel):
    status: str = "ok"
