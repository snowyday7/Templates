#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
user-management - 基于Python后端模板库构建的项目
"""

import os
from pathlib import Path

from templates.api import FastAPIApp, create_fastapi_app
from templates.database import DatabaseManager, DatabaseConfig
from templates.auth import JWTManager, AuthSettings
from templates.monitoring import setup_logging

# 配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")


def create_app():
    """创建应用实例。
    
    Returns:
        应用实例
    """
    # 创建FastAPI应用
    app = create_fastapi_app()
    # 设置数据库配置（需要时再初始化管理器）
    # db_config = DatabaseConfig()
    # db_manager = DatabaseManager(db_config)
    # 使用时: with db_manager.get_session() as session: ...
    # 设置JWT认证
    auth_settings = AuthSettings(SECRET_KEY=SECRET_KEY)
    jwt_manager = JWTManager(auth_settings)
    # 设置日志
    logger = setup_logging()
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
