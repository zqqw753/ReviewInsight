"""并发请求 /api/v1/analysis/analyze，输出延迟分位、RPS 和错误率。

每个请求在问题后追加唯一后缀，绕过 Redis 缓存，测的是完整 Agent 链路
（意图归类 → LLM 决策 → 工具/检索 → 代码拼接）。

示例：
    docker exec -w /app reviewinsight-api-1 python scripts/load_test.py --concurrency 10 --requests 20
"""
import argparse
import asyncio
import random
import time

import httpx

BASE_QUESTIONS = (
    "电池有什么缺点？",
    "外观设计有哪些优点？",
    "这个手机值得买吗？",
    "续航表现怎么样？",
    "发热问题严重吗？",
)


def percentile(values, ratio):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio)))
    return ordered[index]


async def one_request(client, url, index, timeout, run_tag):
    # 每次运行带唯一 tag 绕开 Redis 缓存，测到的是完整链路
    question = f"{random.choice(BASE_QUESTIONS)}（压测 {run_tag}-{index}）"
    started = time.perf_counter()
    try:
        response = await client.post(
            url, json={"question": question}, timeout=timeout
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        body = response.json() if response.status_code == 200 else {}
        meta = body.get("meta") or {}
        ok = response.status_code == 200 and bool(body.get("answer"))
        return {
            "latency_ms": elapsed_ms,
            "ok": ok,
            "status": response.status_code,
            "cached": bool(meta.get("cached")),
            "guarded": bool(meta.get("guarded")),
        }
    except Exception as e:
        return {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "ok": False,
            "status": 0,
            "error": str(e),
        }


async def run(url, concurrency, total, timeout):
    semaphore = asyncio.Semaphore(concurrency)
    run_tag = time.strftime("%H%M%S")

    async with httpx.AsyncClient() as client:
        async def guarded(index):
            async with semaphore:
                return await one_request(client, url, index, timeout, run_tag)

        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(total)))
        wall_seconds = time.perf_counter() - started

    ok_latencies = [item["latency_ms"] for item in results if item["ok"]]
    failures = [item for item in results if not item["ok"]]
    return {
        "concurrency": concurrency,
        "requests": total,
        "wall_seconds": round(wall_seconds, 2),
        "rps": round(total / wall_seconds, 2) if wall_seconds else 0.0,
        "success": len(ok_latencies),
        "failed": len(failures),
        "error_rate": round(len(failures) / total, 3) if total else 0.0,
        "cache_hits": sum(1 for item in results if item.get("cached")),
        "latency_ms": {
            "p50": round(percentile(ok_latencies, 0.5), 1),
            "p95": round(percentile(ok_latencies, 0.95), 1),
            "max": round(max(ok_latencies), 1) if ok_latencies else 0.0,
        },
        "guarded_rate": round(
            sum(1 for item in results if item.get("guarded")) / total, 3
        ) if total else 0.0,
        "first_error": failures[0].get("error", "") if failures else "",
    }


def main():
    parser = argparse.ArgumentParser(description="ReviewInsight /analyze 并发压测")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rounds", type=int, default=1, help="重复轮数，取最后一轮结果")
    args = parser.parse_args()

    url = args.api.rstrip("/") + "/api/v1/analysis/analyze"
    result = None
    for _ in range(args.rounds):
        result = asyncio.run(
            run(url, args.concurrency, args.requests, args.timeout)
        )
    print(result)


if __name__ == "__main__":
    main()
