import hashlib
import json
from typing import Optional
from app.core.redis_client import redis_client
from app.core.config import settings

# key 带上 product_id 和数据版本，数据更新后不会再命中旧答案
def _make_key(question: str, product_id: str = "default", version: str = "") -> str:
    raw = f"{product_id}|{version}|{question}"
    return f"reviewinsight:qa:{hashlib.md5(raw.encode()).hexdigest()}"

async def get_cached_answer(question: str, product_id: str = "default", version: str = "") -> Optional[dict]:
    """返回 {"answer": str, "stats": dict|None}，兼容以前只存纯文本的缓存。"""
    try:
        raw = await redis_client.get(_make_key(question, product_id, version))
    except Exception:
        return None
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"answer": raw, "stats": None}

    if isinstance(data, dict) and "answer" in data:
        return {"answer": data.get("answer", ""), "stats": data.get("stats")}
    return {"answer": raw, "stats": None}

async def set_cached_answer(
    question: str,
    answer: str,
    product_id: str = "default",
    version: str = "",
    stats: Optional[dict] = None,
) -> None:
    """答案和统计一起缓存，命中时接口返回的 stats 和首次一样。"""
    try:
        await redis_client.set(
            _make_key(question, product_id, version),
            json.dumps({"answer": answer, "stats": stats}, ensure_ascii=False),
            ex=settings.redis_ttl
        )
    except Exception:
        pass
