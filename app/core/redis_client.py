import redis.asyncio as redis
from app.core.config import settings

pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    max_connections=20,
    decode_responses=True
)

redis_client = redis.Redis(connection_pool=pool)