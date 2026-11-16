"""Redis 里存最近 N 轮的问题摘要。

只存问题、类型和方面，不存完整答案：改写追问用不到旧答案，
而且旧答案里的数字容易被模型复述，混进本轮代码生成的统计。
"""
import json
from typing import List, Optional

from app.core.redis_client import redis_client
from app.core.config import settings


def _make_key(session_id: str, product_id: str = "default") -> str:
    # 会话按产品隔离，切到别的产品不会串上下文
    return f"reviewinsight:session:{product_id}:{session_id}"


async def load_history(
    session_id: Optional[str],
    product_id: str = "default",
    limit: Optional[int] = None,
) -> List[dict]:
    """返回时间正序的最近 N 轮（最旧在前），供 Agent 解析追问。"""
    if not session_id:
        return []
    size = limit or settings.session_max_turns
    try:
        raw = await redis_client.lrange(_make_key(session_id, product_id), 0, size - 1)
    except Exception:
        # Redis 挂了就退化成单轮问答，会话只是增强
        return []

    turns = []
    for item in raw:
        try:
            data = json.loads(item)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("question"):
            turns.append(data)
    return list(reversed(turns))


async def append_turn(
    session_id: Optional[str],
    question: str,
    question_type: str,
    aspect_label: str = "",
    product_id: str = "default",
) -> None:
    """追加一轮并裁剪到上限；写入失败不影响本次回答。"""
    if not session_id or not question:
        return
    key = _make_key(session_id, product_id)
    record = {
        "question": question,
        "question_type": question_type,
        "aspect_label": aspect_label,
    }
    try:
        await redis_client.lpush(key, json.dumps(record, ensure_ascii=False))
        await redis_client.ltrim(key, 0, settings.session_max_turns - 1)
        await redis_client.expire(key, settings.session_ttl_seconds)
    except Exception:
        pass
