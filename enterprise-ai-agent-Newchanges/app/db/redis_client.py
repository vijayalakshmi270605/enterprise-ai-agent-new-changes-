import logging
import time
import redis.asyncio as redis
from app.config import settings

logger = logging.getLogger(__name__)


class InMemoryRedisBackend:
    def __init__(self):
        self._store = {}
        self._expiry = {}

    def _purge_if_expired(self, key: str):
        expires_at = self._expiry.get(key)
        if expires_at is not None and expires_at <= time.monotonic():
            self._store.pop(key, None)
            self._expiry.pop(key, None)

    async def ping(self):
        return True

    async def close(self):
        return None

    async def get(self, key: str):
        self._purge_if_expired(key)
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int = None):
        self._store[key] = value
        if ex is None:
            self._expiry.pop(key, None)
        else:
            self._expiry[key] = time.monotonic() + ex
        return True

    async def lpush(self, key: str, *values):
        self._purge_if_expired(key)
        current = self._store.get(key)
        if not isinstance(current, list):
            current = []
        for value in values:
            current.insert(0, value)
        self._store[key] = current
        return len(current)

    async def lrange(self, key: str, start: int, end: int):
        self._purge_if_expired(key)
        current = self._store.get(key)
        if not isinstance(current, list):
            return []

        length = len(current)
        if length == 0:
            return []

        if start < 0:
            start = max(length + start, 0)
        if end < 0:
            end = length + end

        if end >= length:
            end = length - 1

        if start > end:
            return []

        return current[start : end + 1]

    async def delete(self, key: str):
        existed = key in self._store
        self._store.pop(key, None)
        self._expiry.pop(key, None)
        return 1 if existed else 0

    async def expire(self, key: str, seconds: int):
        if key not in self._store:
            return False
        self._expiry[key] = time.monotonic() + seconds
        return True


class RedisClient:
    def __init__(self):
        self.redis = None

    async def connect(self):
        try:
            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis memory store")
        except Exception as e:
            logger.info(
                "Redis unavailable at %s; using in-memory fallback.",
                settings.redis_url,
            )
            self.redis = InMemoryRedisBackend()

    async def disconnect(self):
        if self.redis:
            try:
                await self.redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning("Error closing Redis: %s", str(e))

    async def get(self, key: str):
        if not self.redis:
            return None
        return await self.redis.get(key)

    async def set(self, key: str, value: str, ex: int = None):
        if not self.redis:
            return None
        return await self.redis.set(key, value, ex=ex)

    async def lpush(self, key: str, *values):
        if not self.redis:
            return None
        return await self.redis.lpush(key, *values)

    async def lrange(self, key: str, start: int, end: int):
        if not self.redis:
            return []
        return await self.redis.lrange(key, start, end)

    async def delete(self, key: str):
        if not self.redis:
            return None
        return await self.redis.delete(key)

    async def expire(self, key: str, seconds: int):
        if not self.redis:
            return None
        return await self.redis.expire(key, seconds)


redis_client = RedisClient()
