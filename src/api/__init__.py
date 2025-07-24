#!/usr/bin/env python3
"""
API模块初始化文件

导出API相关的组件，包括：
1. 路由器
2. 依赖项
3. 中间件
4. 异常处理器
"""

from .v1 import api_router as v1_router
from .dependencies import *
from .middleware import *
from .exceptions import *

__all__ = [
    "v1_router",
    # 依赖项
    "get_current_user",
    "get_current_active_user",
    "get_current_superuser",
    "require_permissions",
    "require_roles",
    # 中间件
    "setup_api_middleware",
    # 异常处理器
    "setup_api_exception_handlers",
]