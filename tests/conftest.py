import os

# 单测不依赖真实密钥，导入应用前先塞占位配置
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
