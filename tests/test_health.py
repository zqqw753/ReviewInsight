# 依赖不可用时健康检查返回 503，不能一直是 200
from fastapi.testclient import TestClient

import app.main as main


class _OkRedis:
    async def ping(self):
        return True


class _DownRedis:
    async def ping(self):
        raise ConnectionError("redis down")


def test_health_ok(monkeypatch):
    monkeypatch.setattr(main, "redis_client", _OkRedis())
    response = TestClient(main.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_degraded_returns_503(monkeypatch):
    monkeypatch.setattr(main, "redis_client", _DownRedis())
    response = TestClient(main.app).get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "checks": {"redis": False}}


def test_demo_page_is_served():
    # /demo 直接返回单文件页面，不需要前端构建
    response = TestClient(main.app).get("/demo")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ReviewInsight" in response.text
