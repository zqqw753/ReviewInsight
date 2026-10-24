# 使用轻量级 Python 镜像
FROM python:3.11-slim

# 设置环境变量，确保 Python 输出实时打印
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 设置工作目录
WORKDIR /app

# 先复制依赖清单，利用 Docker 缓存加速构建
COPY requirements.txt .

# 配置 pip 清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 再复制项目源码
COPY ./app ./app

# 暴露端口
EXPOSE 8000

# 启动命令（注意 --host 0.0.0.0 是容器外访问的关键）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]