# -*- coding: utf-8 -*-
"""
CORS中间件

提供跨域资源共享(CORS)配置功能。
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# 导入模板库组件
from src.core.logging import get_logger

# 导入应用组件
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class CORSMiddleware(BaseHTTPMiddleware):
    """自定义CORS中间件类"""
    
    def __init__(
        self,
        app,
        allow_origins: List[str] = None,
        allow_credentials: bool = True,
        allow_methods: List[str] = None,
        allow_headers: List[str] = None,
        expose_headers: List[str] = None,
        max_age: int = 600
    ):
        super().__init__(app)
        
        # 设置默认值
        self.allow_origins = allow_origins or settings.CORS_ORIGINS
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods or [
            "GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"
        ]
        self.allow_headers = allow_headers or [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
            "X-API-Key",
            "Cache-Control",
            "Pragma"
        ]
        self.expose_headers = expose_headers or [
            "X-Request-ID",
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ]
        self.max_age = max_age
        
        # 预处理允许的源
        self.processed_origins = self._process_origins(self.allow_origins)
    
    async def dispatch(self, request: Request, call_next):
        """
        中间件处理函数
        """
        origin = request.headers.get("Origin")
        
        # 处理预检请求
        if request.method == "OPTIONS":
            return self._handle_preflight_request(request, origin)
        
        # 处理实际请求
        response = await call_next(request)
        
        # 添加CORS头
        self._add_cors_headers(response, origin)
        
        return response
    
    def _process_origins(self, origins: List[str]) -> List[str]:
        """
        预处理允许的源
        """
        processed = []
        for origin in origins:
            if origin == "*":
                processed.append("*")
            else:
                # 移除尾部斜杠
                processed.append(origin.rstrip("/"))
        return processed
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """
        检查源是否被允许
        """
        if not origin:
            return False
        
        if "*" in self.processed_origins:
            return True
        
        # 移除尾部斜杠进行比较
        origin = origin.rstrip("/")
        
        return origin in self.processed_origins
    
    def _handle_preflight_request(self, request: Request, origin: Optional[str]) -> Response:
        """
        处理预检请求
        """
        headers = {}
        
        # 检查源
        if self._is_origin_allowed(origin):
            if "*" in self.processed_origins and self.allow_credentials:
                # 如果允许凭据，不能使用通配符
                headers["Access-Control-Allow-Origin"] = origin
            else:
                headers["Access-Control-Allow-Origin"] = "*" if "*" in self.processed_origins else origin
        else:
            # 源不被允许，返回403
            logger.warning(f"CORS: Origin not allowed: {origin}")
            return Response(
                content="Origin not allowed",
                status_code=403,
                headers={"Content-Type": "text/plain"}
            )
        
        # 检查请求方法
        requested_method = request.headers.get("Access-Control-Request-Method")
        if requested_method and requested_method not in self.allow_methods:
            logger.warning(f"CORS: Method not allowed: {requested_method}")
            return Response(
                content="Method not allowed",
                status_code=405,
                headers={"Content-Type": "text/plain"}
            )
        
        # 检查请求头
        requested_headers = request.headers.get("Access-Control-Request-Headers")
        if requested_headers:
            requested_headers_list = [h.strip() for h in requested_headers.split(",")]
            for header in requested_headers_list:
                if header.lower() not in [h.lower() for h in self.allow_headers]:
                    logger.warning(f"CORS: Header not allowed: {header}")
                    return Response(
                        content="Header not allowed",
                        status_code=400,
                        headers={"Content-Type": "text/plain"}
                    )
        
        # 设置CORS头
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        headers["Access-Control-Max-Age"] = str(self.max_age)
        
        if self.expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        
        return Response(
            content="",
            status_code=200,
            headers=headers
        )
    
    def _add_cors_headers(self, response: Response, origin: Optional[str]):
        """
        添加CORS头到响应
        """
        if self._is_origin_allowed(origin):
            if "*" in self.processed_origins and self.allow_credentials:
                # 如果允许凭据，不能使用通配符
                response.headers["Access-Control-Allow-Origin"] = origin
            else:
                response.headers["Access-Control-Allow-Origin"] = "*" if "*" in self.processed_origins else origin
            
            if self.allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            if self.expose_headers:
                response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)


class DynamicCORSMiddleware(CORSMiddleware):
    """动态CORS中间件（支持运行时修改配置）"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self._dynamic_origins = set()
    
    def add_origin(self, origin: str):
        """
        动态添加允许的源
        """
        self._dynamic_origins.add(origin.rstrip("/"))
        logger.info(f"CORS: Added dynamic origin: {origin}")
    
    def remove_origin(self, origin: str):
        """
        动态移除允许的源
        """
        origin = origin.rstrip("/")
        if origin in self._dynamic_origins:
            self._dynamic_origins.remove(origin)
            logger.info(f"CORS: Removed dynamic origin: {origin}")
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """
        检查源是否被允许（包括动态添加的源）
        """
        if not origin:
            return False
        
        # 检查静态配置
        if super()._is_origin_allowed(origin):
            return True
        
        # 检查动态添加的源
        origin = origin.rstrip("/")
        return origin in self._dynamic_origins


class SecurityCORSMiddleware(CORSMiddleware):
    """安全增强的CORS中间件"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        
        # 可疑的源模式
        self.suspicious_patterns = [
            "localhost",
            "127.0.0.1",
            "file://",
            "data:",
            "javascript:"
        ]
        
        # 生产环境不允许的源
        self.production_blocked_patterns = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0"
        ]
    
    def _is_origin_allowed(self, origin: Optional[str]) -> bool:
        """
        安全检查源是否被允许
        """
        if not origin:
            return False
        
        # 检查可疑模式
        for pattern in self.suspicious_patterns:
            if pattern in origin.lower():
                logger.warning(f"CORS: Suspicious origin detected: {origin}")
                # 在开发环境可能允许，生产环境拒绝
                if settings.ENVIRONMENT == "production":
                    return False
        
        # 生产环境额外检查
        if settings.ENVIRONMENT == "production":
            for pattern in self.production_blocked_patterns:
                if pattern in origin.lower():
                    logger.warning(f"CORS: Production blocked origin: {origin}")
                    return False
        
        return super()._is_origin_allowed(origin)
    
    def _handle_preflight_request(self, request: Request, origin: Optional[str]) -> Response:
        """
        安全增强的预检请求处理
        """
        # 记录预检请求
        logger.info(f"CORS: Preflight request from origin: {origin}")
        
        # 检查请求频率（防止CORS攻击）
        client_ip = self._get_client_ip(request)
        if self._is_too_frequent_preflight(client_ip):
            logger.warning(f"CORS: Too frequent preflight requests from IP: {client_ip}")
            return Response(
                content="Too many requests",
                status_code=429,
                headers={"Content-Type": "text/plain"}
            )
        
        return super()._handle_preflight_request(request, origin)
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端IP
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_too_frequent_preflight(self, client_ip: str) -> bool:
        """
        检查预检请求是否过于频繁
        """
        # 这里可以实现基于Redis或内存的频率检查
        # 简单实现：每个IP每分钟最多10个预检请求
        # 实际应用中应该使用更复杂的算法
        return False


def create_cors_middleware(
    allow_origins: Optional[List[str]] = None,
    allow_credentials: bool = True,
    allow_methods: Optional[List[str]] = None,
    allow_headers: Optional[List[str]] = None,
    expose_headers: Optional[List[str]] = None,
    max_age: int = 600,
    security_enhanced: bool = True,
    dynamic: bool = False
) -> Union[CORSMiddleware, DynamicCORSMiddleware, SecurityCORSMiddleware]:
    """
    创建CORS中间件实例
    """
    kwargs = {
        "allow_origins": allow_origins,
        "allow_credentials": allow_credentials,
        "allow_methods": allow_methods,
        "allow_headers": allow_headers,
        "expose_headers": expose_headers,
        "max_age": max_age
    }
    
    if dynamic:
        return lambda app: DynamicCORSMiddleware(app, **kwargs)
    elif security_enhanced:
        return lambda app: SecurityCORSMiddleware(app, **kwargs)
    else:
        return lambda app: CORSMiddleware(app, **kwargs)


def get_default_cors_config() -> dict:
    """
    获取默认CORS配置
    """
    return {
        "allow_origins": settings.CORS_ORIGINS,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
        "allow_headers": [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
            "X-API-Key",
            "Cache-Control",
            "Pragma"
        ],
        "expose_headers": [
            "X-Request-ID",
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ],
        "max_age": 600
    }


def setup_fastapi_cors(app, **kwargs):
    """
    使用FastAPI内置CORS中间件设置CORS
    """
    config = get_default_cors_config()
    config.update(kwargs)
    
    app.add_middleware(
        FastAPICORSMiddleware,
        **config
    )
    
    logger.info("CORS middleware configured with FastAPI built-in middleware")