#!/usr/bin/env python3
"""
企业级应用主入口文件

这是应用的主要入口点，负责：
1. 初始化FastAPI应用
2. 配置中间件
3. 注册路由
4. 设置错误处理
5. 配置监控和日志
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.sessions import SessionMiddleware

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import get_settings
from src.core.database import DatabaseManager
from src.core.exceptions import (
    CustomHTTPException,
    ValidationException,
    custom_http_exception_handler,
    validation_exception_handler,
)
from src.core.logging import setup_logging
from src.core.middleware import (
    AuthenticationMiddleware,
    LoggingMiddleware,
    SecurityHeadersMiddleware,
    TimingMiddleware,
)
from src.core.monitoring import setup_monitoring
from src.core.redis import RedisManager
from src.routers import (
    auth_router,
    health_router,
    users_router,
    admin_router,
    api_router,
)

# 获取配置
settings = get_settings()

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 创建限流器
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url,
    default_limits=["1000/hour", "100/minute"]
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    应用生命周期管理
    
    在应用启动时初始化资源，在关闭时清理资源
    """
    logger.info("Starting Enterprise Application...")
    
    try:
        # 初始化数据库连接
        logger.info("Initializing database connection...")
        await DatabaseManager.initialize()
        
        # 初始化Redis连接
        logger.info("Initializing Redis connection...")
        await RedisManager.initialize()
        
        # 设置监控
        if settings.feature_metrics_enabled:
            logger.info("Setting up monitoring...")
            setup_monitoring(app)
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # 清理资源
        logger.info("Shutting down application...")
        
        try:
            await DatabaseManager.close()
            await RedisManager.close()
            logger.info("Application shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """
    创建并配置FastAPI应用实例
    
    Returns:
        FastAPI: 配置好的应用实例
    """
    # 创建FastAPI应用
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.app_debug,
        docs_url="/docs" if settings.feature_api_documentation_enabled else None,
        redoc_url="/redoc" if settings.feature_api_documentation_enabled else None,
        openapi_url="/openapi.json" if settings.feature_api_documentation_enabled else None,
        lifespan=lifespan,
    )
    
    # 配置中间件
    setup_middleware(app)
    
    # 注册路由
    setup_routes(app)
    
    # 设置异常处理器
    setup_exception_handlers(app)
    
    # 挂载静态文件
    setup_static_files(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """
    配置应用中间件
    
    Args:
        app: FastAPI应用实例
    """
    # 安全头中间件
    if settings.security_headers_enabled:
        app.add_middleware(SecurityHeadersMiddleware)
    
    # 受信任主机中间件
    if settings.app_environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # 在生产环境中应该配置具体的主机
        )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # 会话中间件
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        max_age=settings.session_timeout,
        same_site="lax",
        https_only=settings.app_environment == "production",
    )
    
    # 限流中间件
    if settings.feature_rate_limiting_enabled:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)
    
    # Gzip压缩中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 自定义中间件
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthenticationMiddleware)


def setup_routes(app: FastAPI) -> None:
    """
    注册应用路由
    
    Args:
        app: FastAPI应用实例
    """
    # 健康检查路由（不需要认证）
    app.include_router(
        health_router,
        prefix="/health",
        tags=["Health"]
    )
    
    # 认证路由
    app.include_router(
        auth_router,
        prefix="/auth",
        tags=["Authentication"]
    )
    
    # API路由
    app.include_router(
        api_router,
        prefix="/api/v1",
        tags=["API"]
    )
    
    # 用户路由
    app.include_router(
        users_router,
        prefix="/api/v1/users",
        tags=["Users"]
    )
    
    # 管理员路由
    app.include_router(
        admin_router,
        prefix="/api/v1/admin",
        tags=["Admin"]
    )
    
    # Prometheus指标端点
    if settings.feature_metrics_enabled:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)


def setup_exception_handlers(app: FastAPI) -> None:
    """
    设置异常处理器
    
    Args:
        app: FastAPI应用实例
    """
    app.add_exception_handler(CustomHTTPException, custom_http_exception_handler)
    app.add_exception_handler(ValidationException, validation_exception_handler)
    
    @app.exception_handler(500)
    async def internal_server_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        处理内部服务器错误
        """
        logger.error(f"Internal server error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        处理404错误
        """
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": "The requested resource was not found",
                "path": str(request.url.path)
            }
        )


def setup_static_files(app: FastAPI) -> None:
    """
    挂载静态文件
    
    Args:
        app: FastAPI应用实例
    """
    static_dir = Path(settings.static_dir)
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    upload_dir = Path(settings.upload_dir)
    if upload_dir.exists():
        app.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")


# 创建应用实例
app = create_app()


@app.get("/", include_in_schema=False)
async def root() -> dict:
    """
    根路径端点
    
    Returns:
        dict: 应用基本信息
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "environment": settings.app_environment,
        "docs_url": "/docs" if settings.feature_api_documentation_enabled else None,
        "health_url": "/health",
        "metrics_url": "/metrics" if settings.feature_metrics_enabled else None,
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """
    返回空的favicon响应
    """
    return Response(content="", media_type="image/x-icon")


def main() -> None:
    """
    主函数，用于直接运行应用
    """
    if settings.app_environment == "development":
        # 开发环境使用uvicorn直接运行
        uvicorn.run(
            "main:app",
            host=settings.app_host,
            port=settings.app_port,
            reload=settings.app_reload,
            log_level=settings.app_log_level.lower(),
            access_log=True,
        )
    else:
        # 生产环境提示使用gunicorn
        logger.info(
            "For production deployment, use: "
            "gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker "
            f"--bind {settings.app_host}:{settings.app_port}"
        )


if __name__ == "__main__":
    main()