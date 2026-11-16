FROM python:3.11-slim

# 1. 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 2. 安装依赖（配置清华源加速）
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --retries 10 --timeout 120 -r requirements.txt

# 3. 仅模型下载需要镜像源，放在依赖层之后，避免改环境变量导致依赖层缓存失效
ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/opt/hf-cache

# 关键：构建时把模型下载进镜像，运行时就不再拉取 400MB 模型了
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('shibing624/text2vec-base-chinese')"

# 【构建优化】模型层之后立即按数字 UID 授权，避免每次改代码都递归 chown 409MB 缓存
RUN chown -R 10001:10001 /opt/hf-cache

# 4. 复制代码
COPY ./app ./app

# 5. 创建 chroma_db 目录并创建非 root 用户运行（企业级安全规范）
RUN mkdir -p /app/chroma_db
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
# 数据卷挂载点预先创建并授权，避免命名卷以 root 属主创建导致 appuser 无法写入
RUN mkdir -p /data/chroma_db && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000

# 6. 健康检查（使用 Python 内置 urllib 替代 curl）
# 如果 /health 返回 200，正常退出（状态码 0）；如果连接失败或报错，抛出异常退出（状态码 1）
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]