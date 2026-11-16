# 接口契约：stats 随答案透传，空问题直接 400
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


class FakeAgent:
    def invoke(self, state):
        return {
            **state,
            "answer": "根据数据库统计，电池相关提及 12 次",
            "sources": ["sample.txt"],
            "stats": dict(FAKE_STATS),
        }


async def _no_cache(*args, **kwargs):
    return None


async def _no_write(*args, **kwargs):
    return None


def test_analyze_passes_through_stats(monkeypatch):
    monkeypatch.setattr(analysis, "agent_app", FakeAgent())
    monkeypatch.setattr(analysis, "data_version", lambda product_id="default": "1:2026-10-28")
    monkeypatch.setattr(analysis, "get_cached_answer", _no_cache)
    monkeypatch.setattr(analysis, "set_cached_answer", _no_write)

    client = TestClient(app)
    response = client.post("/api/v1/analysis/analyze", json={"question": "电池有什么缺点？"})

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["mentions"] == 12
    assert body["stats"]["negative_ratio"] == 0.75
    assert body["sources"] == ["sample.txt"]
    # meta 随答案返回，缓存没命中时 cached 为 false
    assert body["meta"]["cached"] is False
    assert "latency_ms" in body["meta"]


class FakeResolvedAgent:
    def invoke(self, state):
        return {
            **state,
            "answer": "根据数据库统计，电池相关提及 12 次",
            "sources": ["sample.txt"],
            "stats": dict(FAKE_STATS),
            "question_type": "优点",
            "resolved_question": "电池有什么优点？",
        }


def test_session_skips_cache_and_records_turn(monkeypatch):
    # 带 session_id 时不碰单轮缓存，改成写会话历史，存的是改写后的问题
    calls = {"cache": 0, "append": []}

    async def _cache(*args, **kwargs):
        calls["cache"] += 1
        return None

    async def _append(*args, **kwargs):
        calls["append"].append(args)

    async def _history(*args, **kwargs):
        return [{"question": "电池有什么缺点？", "question_type": "缺点", "aspect_label": "电池"}]

    monkeypatch.setattr(analysis, "agent_app", FakeResolvedAgent())
    monkeypatch.setattr(analysis, "data_version", lambda product_id="default": "1:2026-10-28")
    monkeypatch.setattr(analysis, "get_cached_answer", _cache)
    monkeypatch.setattr(analysis, "set_cached_answer", _cache)
    monkeypatch.setattr(analysis, "load_history", _history)
    monkeypatch.setattr(analysis, "append_turn", _append)

    client = TestClient(app)
    response = client.post(
        "/api/v1/analysis/analyze",
        json={"question": "那优点呢", "session_id": "s-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["cached"] is False
    assert body["meta"]["memory_turns"] == 1
    assert calls["cache"] == 0
    assert calls["append"][0][0] == "s-1"
    assert calls["append"][0][1] == "电池有什么优点？"


def test_metrics_endpoint(monkeypatch):
    # /metrics 返回进程指标和索引条数，多次请求会累加
    from app.core import metrics as core_metrics
    import app.routers.metrics as metrics_router

    core_metrics.reset()
    monkeypatch.setattr(metrics_router, "index_counts", lambda: {"reviews": 2, "vectors": 2})

    client = TestClient(app)
    body = client.get("/metrics").json()

    assert body["requests"] == 0
    assert body["index"] == {"reviews": 2, "vectors": 2}
    for key in ("errors", "cache_hits", "guardrail_hits", "input_tokens", "output_tokens"):
        assert key in body


def test_empty_question_rejected(monkeypatch):
    monkeypatch.setattr(analysis, "agent_app", FakeAgent())
    client = TestClient(app)
    response = client.post("/api/v1/analysis/analyze", json={"question": "   "})
    assert response.status_code == 400


async def _cached_with_stats(*args, **kwargs):
    # 模拟命中缓存，缓存值里也存了 stats
    return {"answer": "根据数据库统计，电池相关提及 12 次", "stats": dict(FAKE_STATS)}


def test_cached_answer_keeps_stats(monkeypatch):
    monkeypatch.setattr(analysis, "data_version", lambda product_id="default": "1:2026-10-28")
    monkeypatch.setattr(analysis, "get_cached_answer", _cached_with_stats)
    monkeypatch.setattr(analysis, "agent_app", FakeAgent())

    client = TestClient(app)
    response = client.post("/api/v1/analysis/analyze", json={"question": "电池有什么缺点？"})

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["mentions"] == 12
    assert body["stats"]["negative_ratio"] == 0.75

