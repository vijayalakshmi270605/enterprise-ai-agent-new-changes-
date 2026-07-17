import json
import logging
from app.db.redis_client import redis_client
from app.config import settings

logger = logging.getLogger(__name__)


class MemoryService:
    SESSION_PREFIX = "assistant:session:"

    @classmethod
    async def _get_key(cls, session_id: str) -> str:
        return f"{cls.SESSION_PREFIX}{session_id}"

    @classmethod
    async def append_user_message(cls, session_id: str, content: str):
        key = await cls._get_key(session_id)
        record = json.dumps({"role": "user", "content": content})
        await redis_client.lpush(key, record)
        await redis_client.expire(key, 60 * 60 * 24)
        logger.debug("Appended user message to %s", session_id)

    @classmethod
    async def append_assistant_message(cls, session_id: str, content: str):
        key = await cls._get_key(session_id)
        record = json.dumps({"role": "assistant", "content": content})
        await redis_client.lpush(key, record)
        await redis_client.expire(key, 60 * 60 * 24)
        logger.debug("Appended assistant message to %s", session_id)

    @classmethod
    async def get_history(cls, session_id: str, limit: int = None):
        key = await cls._get_key(session_id)
        limit = limit or settings.max_history
        records = await redis_client.lrange(key, 0, limit - 1)
        messages = [json.loads(item) for item in reversed(records)]
        return messages

    @classmethod
    async def clear_session(cls, session_id: str):
        key = await cls._get_key(session_id)
        await redis_client.delete(key)
        logger.info("Cleared session memory %s", session_id)


memory_service = MemoryService()
