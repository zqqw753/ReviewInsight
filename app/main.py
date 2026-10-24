from fastapi import FastAPI
from app.core.config import settings
from app.routers import analysis

app = FastAPI(
    title="ReviewInsight",
    description="跨境电商AI选品分析系统",
    version="0.1.0"
)

app.include_router(analysis.router)

@app.get("/")
async def root():
    return {"message": "ReviewInsight is running"}

@app.get("/health/config")
async def check_config():
    return {
        "status": "ok",
        "chroma_dir": settings.chroma_persist_dir,
        "api_key_loaded": bool(settings.deepseek_api_key)
    }