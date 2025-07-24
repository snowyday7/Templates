# 企业级Python应用Docker镜像
# 多阶段构建，优化镜像大小和安全性

# =============================================================================
# 构建阶段
# =============================================================================
FROM python:3.11-slim as builder

# 设置构建参数
ARG BUILD_ENV=production
ARG APP_VERSION=1.0.0

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建应用用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt requirements-prod.txt ./

# 安装Python依赖
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "$BUILD_ENV" = "production" ]; then \
        pip install --no-cache-dir -r requirements-prod.txt; \
    fi

# =============================================================================
# 运行时阶段
# =============================================================================
FROM python:3.11-slim as runtime

# 设置标签
LABEL maintainer="Enterprise Team <team@enterprise.com>" \
      version="${APP_VERSION}" \
      description="Enterprise Python Application" \
      org.opencontainers.image.source="https://github.com/enterprise/templates"

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    APP_ENV=production \
    APP_VERSION=${APP_VERSION}

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    libpq5 \
    libssl3 \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建应用用户和目录
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app /app/logs /app/data && \
    chown -R appuser:appuser /app

# 从构建阶段复制Python包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY --chown=appuser:appuser . .

# 创建必要的目录
RUN mkdir -p logs data temp && \
    chown -R appuser:appuser logs data temp

# 切换到应用用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 设置启动命令
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["python", "-m", "gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--preload"]

# =============================================================================
# 开发环境镜像
# =============================================================================
FROM runtime as development

# 切换回root用户安装开发依赖
USER root

# 安装开发工具
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 安装开发依赖
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# 切换回应用用户
USER appuser

# 开发环境使用不同的启动命令
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]

# =============================================================================
# 测试环境镜像
# =============================================================================
FROM development as testing

# 复制测试文件
COPY --chown=appuser:appuser tests/ tests/
COPY --chown=appuser:appuser pytest.ini .
COPY --chown=appuser:appuser .coveragerc .

# 运行测试
RUN python -m pytest tests/ --cov=templates --cov-report=html --cov-report=term

# 测试环境启动命令
CMD ["python", "-m", "pytest", "tests/", "-v"]