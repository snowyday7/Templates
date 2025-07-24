# -*- coding: utf-8 -*-
"""
主应用文件

整合所有组件并启动FastAPI应用。
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# 导入模板库组件
from src.core.logging import get_logger, setup_logging
from src.middleware.request_id import RequestIDMiddleware
from src.middleware.timing import TimingMiddleware

# 导入应用组件
from app.core.config import get_settings
from app.core.database import init_database, get_database
from app.core.cache import get_cache
from app.middleware import (
    AuthMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    CORSMiddleware as AppCORSMiddleware
)
from app.exceptions.handlers import setup_exception_handlers
from app.api.v1 import auth, conversations, messages, users, websocket

# 设置日志
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    # 启动时执行
    logger.info("Starting ChatGPT Backend application...")
    
    try:
        # 初始化数据库
        init_database()
        logger.info("Database initialized")
        
        # 初始化缓存
        cache = get_cache()
        cache_info = cache.get_info()
        logger.info(f"Cache initialized: {cache_info['type']} (available: {cache_info['available']})")
        
        # 其他初始化任务
        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("Shutting down ChatGPT Backend application...")
    
    try:
        # 关闭数据库连接
        database = get_database()
        database.close()
        logger.info("Database connections closed")
        
        # 清理缓存
        cache = get_cache()
        cache.cleanup()
        logger.info("Cache cleaned up")
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error(f"Application shutdown error: {e}")


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan
    )
    
    # 设置中间件
    setup_middleware(app)
    
    # 设置异常处理器
    setup_exception_handlers(app)
    
    # 设置路由
    setup_routes(app)
    
    # 设置健康检查和监控端点
    setup_monitoring(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """
    设置中间件
    """
    # 请求ID中间件（最先添加）
    app.add_middleware(RequestIDMiddleware)
    
    # 时间统计中间件
    app.add_middleware(TimingMiddleware)
    
    # 请求日志中间件
    if settings.LOG_REQUESTS:
        app.add_middleware(
            RequestLoggingMiddleware,
            include_request_body=settings.LOG_RESPONSES,
            include_response_body=settings.LOG_RESPONSES,
            max_body_size=1024 * 10,  # 10KB
            skip_paths=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        )
    
    # 认证中间件
    app.add_middleware(AuthMiddleware)
    
    # 限流中间件
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        requests_per_hour=settings.RATE_LIMIT_REQUESTS_PER_HOUR,
        requests_per_day=settings.RATE_LIMIT_REQUESTS_PER_DAY
    )
    
    # CORS中间件
    if settings.ENABLE_CORS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=settings.CORS_METHODS,
            allow_headers=settings.CORS_HEADERS,
        )
    
    # 受信任主机中间件
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
    
    # Gzip压缩中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    logger.info("Middleware setup completed")


def setup_routes(app: FastAPI) -> None:
    """
    设置路由
    """
    # API v1 路由
    app.include_router(
        auth.router,
        prefix=f"{settings.API_V1_STR}/auth",
        tags=["认证"]
    )
    
    app.include_router(
        users.router,
        prefix=f"{settings.API_V1_STR}/users",
        tags=["用户管理"]
    )
    
    app.include_router(
        conversations.router,
        prefix=f"{settings.API_V1_STR}/conversations",
        tags=["对话管理"]
    )
    
    app.include_router(
        messages.router,
        prefix=f"{settings.API_V1_STR}/messages",
        tags=["消息管理"]
    )
    
    app.include_router(
        websocket.router,
        prefix=f"{settings.API_V1_STR}/ws",
        tags=["WebSocket"]
    )
    
    logger.info("Routes setup completed")


def setup_monitoring(app: FastAPI) -> None:
    """
    设置监控端点
    """
    
    @app.get("/health", tags=["监控"])
    async def health_check():
        """
        健康检查端点
        """
        database = get_database()
        cache = get_cache()
        
        db_healthy = database.health_check()
        cache_healthy = cache.health_check()
        
        status = "healthy" if db_healthy and cache_healthy else "unhealthy"
        
        return {
            "status": status,
            "timestamp": "2024-01-01T00:00:00Z",  # 实际应用中使用 datetime.utcnow().isoformat()
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "services": {
                "database": "healthy" if db_healthy else "unhealthy",
                "cache": "healthy" if cache_healthy else "unhealthy"
            }
        }
    
    @app.get("/info", tags=["监控"])
    async def app_info():
        """
        应用信息端点
        """
        database = get_database()
        cache = get_cache()
        
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": settings.APP_DESCRIPTION,
            "environment": settings.ENVIRONMENT,
            "database": database.get_connection_info(),
            "cache": cache.get_info(),
            "features": {
                "cors_enabled": settings.ENABLE_CORS,
                "metrics_enabled": settings.ENABLE_METRICS,
                "auto_backup_enabled": settings.ENABLE_AUTO_BACKUP
            }
        }
    
    if settings.ENABLE_METRICS:
        @app.get("/metrics", tags=["监控"])
        async def metrics():
            """
            指标端点（Prometheus格式）
            """
            # 这里可以集成Prometheus客户端库
            # 目前返回简单的指标信息
            database = get_database()
            cache = get_cache()
            
            metrics_data = {
                "app_info": {
                    "name": settings.APP_NAME,
                    "version": settings.APP_VERSION,
                    "environment": settings.ENVIRONMENT
                },
                "database_status": 1 if database.health_check() else 0,
                "cache_status": 1 if cache.health_check() else 0,
                "uptime_seconds": 0  # 实际应用中计算运行时间
            }
            
            return metrics_data
    
    # 根路径重定向到文档
    @app.get("/", include_in_schema=False)
    async def root():
        """
        根路径
        """
        if settings.ENVIRONMENT != "production":
            return {"message": f"Welcome to {settings.APP_NAME} API", "docs": "/docs"}
        else:
            return {"message": f"{settings.APP_NAME} API is running"}
    
    logger.info("Monitoring endpoints setup completed")


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    # 开发环境直接运行
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.LOG_REQUESTS,
        workers=1 if settings.ENVIRONMENT == "development" else settings.WORKERS
    )