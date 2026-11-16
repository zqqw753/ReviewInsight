# 流式接口按节点推进度，最后一帧带答案和统计
import json

from fastapi.testclient import TestClient

import app.routers.analysis as analysis
from app.main import app

FAKE_STATS = {
    "aspect": "battery",
    "label": "电池",
    "mentions": 12,
    "negative": 9,
    "negative_ratio": 0.75,
    "total_reviews_in_db": 30,
}


class FakeStreamAgent:
    async def astream(self, state, stream_mode="updates"):
        yield {"classify_question": {
            "question_type": "缺点",
            "trace": [{"node": "classify_question", "seconds": 0.01}],
        }}
        yield {"tools": {
            "messages": [],
            "tool_rounds": 1,
            "trace": [{"node": "tools", "seconds": 0.02}],
        }}
        yield {"finalize": {
            "answer": "根据数据库统计，电池相关提及 12 次",
            "stats": dict(FAKE_STATS),
            "sources": ["sample.txt"],
            "trace": [{"node": "finalize", "seconds": 0.03}],
        }}


async def _no_cache(*args, **kwargs):
    return None


async def _no_write(*args, **kwargs):
    return None


def test_stream_emits_nodes_and_answer(monkeypatch):
    monkeypatch.setattr(analysis, "agent_app", FakeStreamAgent())
    monkeypatch.setattr(analysis, "data_version", lambda product_id="default": "1:2026-10-31")
    monkeypatch.setattr(analysis, "get_cached_answer", _no_cache)
    monkeypatch.setattr(analysis, "set_cached_answer", _no_write)

    client = TestClient(app)
    with client.stream(
        "POST", "/api/v1/analysis/analyze/stream",
        json={"question": "电池有什么缺点？"},
    ) as response:
        body = "".join(response.iter_text())
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    assert "event: node" in body
    assert '"node": "classify_question"' in body
    assert '"node": "finalize"' in body
    assert '"answer":' in body
    assert "event: done" in body

    # 流式 finalize 的 meta 要把各节点 trace 和轮次汇总到一起
    finalize_frame = next(
        block for block in body.split("\n\n") if '"node": "finalize"' in block
    )
    payload = json.loads(finalize_frame.split("data: ", 1)[1].strip())
    meta = payload["meta"]
    assert meta["tool_rounds"] == 1
    assert [item["node"] for item in meta["trace"]] == [
        "classify_question", "tools", "finalize",
    ]
