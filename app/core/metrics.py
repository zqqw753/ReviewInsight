"""进程内指标统计，记录请求量、错误、缓存命中和 token 用量。

单进程 uvicorn 够用，多副本部署要换成 Prometheus 或 Redis 计数器。
"""
import threading

_lock = threading.Lock()
_state = {
    "requests": 0,
    "errors": 0,
    "cache_hits": 0,
    "guardrail_hits": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "latency_ms_total": 0.0,
}


def record_request(
    latency_ms: float,
    *,
    error: bool = False,
    cached: bool = False,
    guardrail: bool = False,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    with _lock:
        _state["requests"] += 1
        _state["latency_ms_total"] += max(0.0, latency_ms)
        _state["errors"] += 1 if error else 0
        _state["cache_hits"] += 1 if cached else 0
        _state["guardrail_hits"] += 1 if guardrail else 0
        _state["input_tokens"] += input_tokens
        _state["output_tokens"] += output_tokens


def snapshot() -> dict:
    with _lock:
        data = dict(_state)
    requests = data["requests"]
    data["avg_latency_ms"] = round(data["latency_ms_total"] / requests, 1) if requests else 0.0
    data["latency_ms_total"] = round(data["latency_ms_total"], 1)
    return data


def reset() -> None:
    """测试用，清空计数，免得用例之间互相影响。"""
    with _lock:
        for key in _state:
            _state[key] = 0
