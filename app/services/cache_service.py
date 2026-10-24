import hashlib
from typing import Optional
from app.core.redis_client import redis_client
from app.core.config import settings

def _make_key(question: str) -> str:
    return f"reviewinsight:qa:{hashlib.md5(question.encode()).hexdigest()}"

async def get_cached_answer(question: str) -> Optional[str]:
    try:
        return await redis_client.get(_make_key(question))
    except Exception:
        return None

async def set_cached_answer(question: str, answer: str) -> None:
    try:
        await redis_client.set(
            _make_key(question),
            answer,
            ex=settings.redis_ttl
        )
    except Exception:
        pass
