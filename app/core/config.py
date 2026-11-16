from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 读取项目根目录的 .env，环境变量名不区分大小写
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    deepseek_api_key: str
    chroma_persist_dir: str = "./chroma_db"
    db_path: str = "./data/reviews.db"
    # 指纹日志路径可配，容器里指到持久化卷，重建容器不丢
    processed_files_log: str = "./processed_files.json"
    log_level: str = "INFO"
    DEFAULT_RETRIEVAL_K: int = 10
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 86400
    # 轮次上限和超时，防止 Agent 一直循环或请求挂着
    max_tool_rounds: int = 4
    llm_timeout_seconds: float = 60.0
    # 只留最近 N 轮问题摘要，过期自动消失
    session_max_turns: int = 3
    session_ttl_seconds: int = 1800

# 单例模式：全局只加载一次
settings = Settings()