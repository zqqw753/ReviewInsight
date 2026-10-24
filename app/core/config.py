import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    deepseek_api_key: str
    chroma_persist_dir: str = "./chroma_db"
    log_level: str = "INFO"
    DEFAULT_RETRIEVAL_K: int = 10
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 86400

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False   # 允许环境变量大小写混用

# 单例模式：全局只加载一次
settings = Settings()