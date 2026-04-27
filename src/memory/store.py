import json
import redis.asyncio as redis
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    messages_to_dict,
    messages_from_dict,
)
from src.config import settings

_client: redis.Redis | None = None
SESSION_TTL = 1800  # 30 minutes
MAX_MESSAGES = 20


async def init_redis() -> redis.Redis:
    global _client
    _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client


async def get_redis() -> redis.Redis:
    if _client is None:
        return await init_redis()
    return _client


async def close_redis():
    global _client
    if _client:
        await _client.aclose()
        _client = None


def _key(session_id: str) -> str:
    return f"session:{session_id}"


async def load_history(session_id: str) -> list[BaseMessage]:
    r = await get_redis()
    data = await r.get(_key(session_id))
    if not data:
        return []
    await r.expire(_key(session_id), SESSION_TTL)
    dicts = json.loads(data)
    return messages_from_dict(dicts)


async def save_history(session_id: str, messages: list[BaseMessage]):
    r = await get_redis()
    trimmed = messages[-MAX_MESSAGES:]
    data = json.dumps(messages_to_dict(trimmed))
    await r.set(_key(session_id), data, ex=SESSION_TTL)
