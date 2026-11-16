import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from app.routers import analysis, metrics
from app.core.config import settings
from app.core.redis_client import redis_client

# 日志级别从配置读，改环境变量就能调
logging.basicConfig(
    level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

# 配置日志，用于记录探活失败的原因
logger = logging.getLogger(__name__)

# 演示页是单个 html，不用前端构建，随 app 目录一起进镜像
DEMO_PAGE = Path(__file__).resolve().parent / "static" / "index.html"

app = FastAPI(
    title="ReviewInsight",
    description="跨境电商AI选品分析系统",
    version="0.2.0"
)

app.include_router(analysis.router)
# /metrics 暴露请求量、错误、缓存命中和 token 用量
app.include_router(metrics.router)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "ReviewInsight is running", "demo": "/demo", "docs": "/docs"}

@app.get("/demo", include_in_schema=False)
async def demo():
    return FileResponse(DEMO_PAGE)

@app.get("/health")
async def health():
    checks = {"redis": False}
    try:
        # 少了 await 拿到的是协程对象，ping 不会真正执行
        checks["redis"] = await redis_client.ping()
    except Exception as e:
        # 打一条 warning 留痕，对外仍然返回 degraded
        logger.warning(f"Health check: Redis unreachable - {str(e)}")
        checks["redis"] = False

    ok = all(checks.values())
    # 依赖挂了要返回 503，Docker healthcheck 和调用方才看得出降级
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ok" if ok else "degraded",
            "checks": checks
        }
    )