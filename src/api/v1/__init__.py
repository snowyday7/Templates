#!/usr/bin/env python3
"""
API v1版本模块

定义API v1版本的路由和端点
"""

from fastapi import APIRouter

from .auth import router as auth_router
from .users import router as users_router
from .admin import router as admin_router
from .health import router as health_router

# 创建API v1路由器
api_router = APIRouter(prefix="/api/v1")

# 注册子路由
api_router.include_router(
    health_router,
    prefix="/health",
    tags=["健康检查"]
)

api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["认证"]
)

api_router.include_router(
    users_router,
    prefix="/users",
    tags=["用户管理"]
)

api_router.include_router(
    admin_router,
    prefix="/admin",
    tags=["系统管理"]
)

__all__ = ["api_router"]