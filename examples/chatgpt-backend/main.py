#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGPT 客户端后端项目主入口

这个文件展示了如何使用模板库中的组件快速搭建一个生产级的ChatGPT后端服务。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径，以便导入模板库
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

# 导入模板库组件
from src.core.config import Settings
from src.core.database import DatabaseManager
from src.core.cache import CacheManager
from src.core.logging import setup_logging
from src.core.middleware import (
    RequestLoggingMiddleware,
    SecurityMiddleware,
    RateLimitMiddleware
)
from src.core.exceptions import (
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    RateLimitException
)
from src.utils.response import create_error_response

# 导入应用模块
from app.core.config import get_settings
from app.core.database import init_database
from app.api.v1 import auth, conversations, messages, users
from app.api.websocket import websocket_router
from app.core.openai_client import OpenAIClient

# 全局设置
settings = get_settings()

# 设置日志
logger = setup_logging(
    level=settings.log_level,
    format_type=settings.log_format,
    log_file=settings.log_file
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("启动 ChatGPT 后端服务...")
    
    # 初始化数据库
    await init_database()
    logger.info("数据库初始化完成")
    
    # 初始化缓存
    cache_manager = CacheManager(
        redis_url=settings.redis_url,
        password=settings.redis_password
    )
    await cache_manager.connect()
    app.state.cache = cache_manager
    logger.info("缓存连接建立")
    
    # 初始化OpenAI客户端
    openai_client = OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        max_tokens=settings.openai_max_tokens,
        temperature=settings.openai_temperature
    )
    app.state.openai = openai_client
    logger.info("OpenAI客户端初始化完成")
    
    logger.info(f"服务启动完成，监听 {settings.app_host}:{settings.app_port}")
    
    yield
    
    # 清理资源
    logger.info("正在关闭服务...")
    if hasattr(app.state, 'cache'):
        await app.state.cache.disconnect()
        logger.info("缓存连接已关闭")
    
    logger.info("服务已关闭")


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="基于模板库构建的ChatGPT客户端后端API",
        docs_url="/docs" if settings.app_debug else None,
        redoc_url="/redoc" if settings.app_debug else None,
        lifespan=lifespan
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 自定义中间件（使用模板库组件）
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_requests_per_minute,
        burst=settings.rate_limit_burst
    )
    
    # 异常处理器
    @app.exception_handler(ValidationException)
    async def validation_exception_handler(request: Request, exc: ValidationException):
        return JSONResponse(
            status_code=400,
            content=create_error_response(
                message=str(exc),
                error_code="VALIDATION_ERROR",
                details=exc.details if hasattr(exc, 'details') else None
            )
        )
    
    @app.exception_handler(AuthenticationException)
    async def authentication_exception_handler(request: Request, exc: AuthenticationException):
        return JSONResponse(
            status_code=401,
            content=create_error_response(
                message="认证失败",
                error_code="AUTHENTICATION_ERROR"
            )
        )
    
    @app.exception_handler(AuthorizationException)
    async def authorization_exception_handler(request: Request, exc: AuthorizationException):
        return JSONResponse(
            status_code=403,
            content=create_error_response(
                message="权限不足",
                error_code="AUTHORIZATION_ERROR"
            )
        )
    
    @app.exception_handler(RateLimitException)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
        return JSONResponse(
            status_code=429,
            content=create_error_response(
                message="请求过于频繁，请稍后再试",
                error_code="RATE_LIMIT_ERROR"
            )
        )
    
    # 健康检查端点
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.app_version
        }
    
    # 注册API路由
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["认证"])
    app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["对话"])
    app.include_router(messages.router, prefix="/api/v1/messages", tags=["消息"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["用户"])
    
    # WebSocket路由
    app.include_router(websocket_router, prefix="/api/v1/ws")
    
    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    # 开发模式启动
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )