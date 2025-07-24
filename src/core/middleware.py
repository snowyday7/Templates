#!/usr/bin/env python3
"""
中间件模块

提供各种中间件功能，包括：
1. 请求ID追踪
2. 请求日志记录
3. 性能监控
4. 安全头设置
5. CORS处理
6. 限流控制
7. 异常捕获
8. 请求验证
"""

import time
import uuid
import json
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware as StarletteBaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp
import httpx

from .config import get_settings
from .logging import get_logger
from .security import SecurityHeaders, RateLimiter
from .exceptions import RateLimitException


# 配置日志
logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    请求ID中间件
    
    为每个请求生成唯一ID，用于追踪和日志关联
    """
    
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成或获取请求ID
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        
        # 将请求ID存储到请求状态中
        request.state.request_id = request_id
        
        # 处理请求
        response = await call_next(request)
        
        # 在响应头中添加请求ID
        response.headers[self.header_name] = request_id
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    
    记录请求和响应的详细信息
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or []
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查是否需要排除此路径
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', None)
        
        # 记录请求信息
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # 记录请求体（如果启用）
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # 尝试解析JSON
                    try:
                        request_info["body"] = json.loads(body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        request_info["body"] = body.decode("utf-8", errors="ignore")[:1000]
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")
        
        logger.info("Request started", extra=request_info)
        
        # 处理请求
        try:
            response = await call_next(request)
        except Exception as e:
            # 记录异常
            process_time = time.time() - start_time
            error_info = request_info.copy()
            error_info.update({
                "status_code": 500,
                "process_time": process_time,
                "error": str(e),
                "error_type": type(e).__name__,
            })
            logger.error("Request failed", extra=error_info)
            raise
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录响应信息
        response_info = request_info.copy()
        response_info.update({
            "status_code": response.status_code,
            "process_time": process_time,
            "response_headers": dict(response.headers),
        })
        
        # 记录响应体（如果启用）
        if self.log_response_body and hasattr(response, 'body'):
            try:
                if hasattr(response.body, 'decode'):
                    body_text = response.body.decode("utf-8", errors="ignore")[:1000]
                    try:
                        response_info["response_body"] = json.loads(body_text)
                    except json.JSONDecodeError:
                        response_info["response_body"] = body_text
            except Exception as e:
                logger.warning(f"Failed to read response body: {e}")
        
        # 根据状态码选择日志级别
        if response.status_code >= 500:
            logger.error("Request completed with server error", extra=response_info)
        elif response.status_code >= 400:
            logger.warning("Request completed with client error", extra=response_info)
        else:
            logger.info("Request completed successfully", extra=response_info)
        
        return response
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """获取客户端IP地址"""
        # 检查代理头
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # 返回直接连接的IP
        return request.client.host if request.client else None


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    性能监控中间件
    
    监控请求处理时间和系统性能指标
    """
    
    def __init__(
        self,
        app: ASGIApp,
        slow_request_threshold: float = 1.0,
        enable_metrics: bool = True,
    ):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.enable_metrics = enable_metrics
        self.request_count = 0
        self.total_time = 0.0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 更新统计信息
        if self.enable_metrics:
            self.request_count += 1
            self.total_time += process_time
        
        # 添加性能头
        response.headers["X-Process-Time"] = str(round(process_time, 4))
        
        # 记录慢请求
        if process_time > self.slow_request_threshold:
            request_id = getattr(request.state, 'request_id', None)
            logger.warning(
                f"Slow request detected: {process_time:.4f}s",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "process_time": process_time,
                    "status_code": response.status_code,
                }
            )
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if self.request_count == 0:
            return {
                "request_count": 0,
                "average_response_time": 0.0,
                "total_time": 0.0,
            }
        
        return {
            "request_count": self.request_count,
            "average_response_time": self.total_time / self.request_count,
            "total_time": self.total_time,
        }


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    安全头中间件
    
    添加各种安全相关的HTTP头
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security_headers = SecurityHeaders()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # 添加安全头
        headers = self.security_headers.get_headers()
        for name, value in headers.items():
            response.headers[name] = value
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    限流中间件
    
    基于IP地址或用户ID进行请求限流
    """
    
    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: RateLimiter,
        exclude_paths: Optional[List[str]] = None,
        get_user_id: Optional[Callable[[Request], Optional[str]]] = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.exclude_paths = exclude_paths or []
        self.get_user_id = get_user_id
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查是否需要排除此路径
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # 获取限流键
        if self.get_user_id:
            user_id = self.get_user_id(request)
            key = f"user:{user_id}" if user_id else f"ip:{self._get_client_ip(request)}"
        else:
            key = f"ip:{self._get_client_ip(request)}"
        
        # 检查限流
        allowed, retry_after = await self.rate_limiter.is_allowed(key)
        
        if not allowed:
            raise RateLimitException(
                message="Rate limit exceeded",
                retry_after=retry_after
            )
        
        response = await call_next(request)
        
        # 添加限流信息到响应头
        remaining = await self.rate_limiter.get_remaining(key)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        if retry_after:
            response.headers["X-RateLimit-Reset"] = str(retry_after)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    健康检查中间件
    
    提供简单的健康检查端点
    """
    
    def __init__(
        self,
        app: ASGIApp,
        health_path: str = "/health",
        readiness_path: str = "/ready",
    ):
        super().__init__(app)
        self.health_path = health_path
        self.readiness_path = readiness_path
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path == self.health_path:
            return self._health_response()
        elif request.url.path == self.readiness_path:
            return await self._readiness_response()
        
        return await call_next(request)
    
    def _health_response(self) -> Response:
        """健康检查响应"""
        return Response(
            content=json.dumps({
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
            }),
            media_type="application/json"
        )
    
    async def _readiness_response(self) -> Response:
        """就绪检查响应"""
        # 这里可以添加更复杂的就绪检查逻辑
        # 例如检查数据库连接、缓存连接等
        
        try:
            # 简单的就绪检查
            ready = True
            checks = {
                "database": "ok",  # 实际应用中应该检查数据库连接
                "cache": "ok",     # 实际应用中应该检查缓存连接
            }
            
            status_code = 200 if ready else 503
            
            return Response(
                content=json.dumps({
                    "status": "ready" if ready else "not ready",
                    "checks": checks,
                    "timestamp": datetime.utcnow().isoformat(),
                }),
                status_code=status_code,
                media_type="application/json"
            )
        except Exception as e:
            return Response(
                content=json.dumps({
                    "status": "not ready",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }),
                status_code=503,
                media_type="application/json"
            )


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    请求大小限制中间件
    
    限制请求体的大小
    """
    
    def __init__(self, app: ASGIApp, max_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查Content-Length头
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            return Response(
                content=json.dumps({
                    "error": "Request entity too large",
                    "max_size": self.max_size,
                }),
                status_code=413,
                media_type="application/json"
            )
        
        return await call_next(request)


# =============================================================================
# 中间件配置函数
# =============================================================================

def setup_middleware(app, settings=None):
    """
    设置中间件
    
    Args:
        app: FastAPI应用实例
        settings: 配置对象
    """
    if settings is None:
        settings = get_settings()
    
    # 1. 请求大小限制（最先执行）
    app.add_middleware(
        RequestSizeMiddleware,
        max_size=settings.max_request_size
    )
    
    # 2. 健康检查
    app.add_middleware(HealthCheckMiddleware)
    
    # 3. 安全头
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 4. CORS（如果启用）
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=settings.cors_allow_methods,
            allow_headers=settings.cors_allow_headers,
            expose_headers=settings.cors_expose_headers,
            max_age=settings.cors_max_age,
        )
    
    # 5. 受信任主机（如果配置）
    if settings.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts
        )
    
    # 6. Gzip压缩
    if settings.enable_gzip:
        app.add_middleware(
            GZipMiddleware,
            minimum_size=settings.gzip_minimum_size
        )
    
    # 7. 限流（如果启用）
    if settings.rate_limit_enabled:
        from .cache import CacheManager
        cache_manager = CacheManager()
        rate_limiter = RateLimiter(
            cache_manager=cache_manager,
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window,
        )
        
        app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=rate_limiter,
            exclude_paths=["/health", "/ready", "/metrics"]
        )
    
    # 8. 性能监控
    app.add_middleware(
        PerformanceMiddleware,
        slow_request_threshold=settings.slow_request_threshold,
        enable_metrics=True
    )
    
    # 9. 请求日志
    app.add_middleware(
        RequestLoggingMiddleware,
        log_request_body=settings.log_request_body,
        log_response_body=settings.log_response_body,
        exclude_paths=["/health", "/ready", "/metrics"]
    )
    
    # 10. 请求ID（最后执行，最先生效）
    app.add_middleware(RequestIDMiddleware)
    
    logger.info("Middleware setup completed")


# =============================================================================
# 中间件管理器
# =============================================================================

class MiddlewareManager:
    """
    中间件管理器
    
    统一管理所有中间件的配置和状态
    """
    
    def __init__(self):
        self.performance_middleware: Optional[PerformanceMiddleware] = None
        self.rate_limit_middleware: Optional[RateLimitMiddleware] = None
    
    def register_performance_middleware(self, middleware: PerformanceMiddleware):
        """注册性能中间件"""
        self.performance_middleware = middleware
    
    def register_rate_limit_middleware(self, middleware: RateLimitMiddleware):
        """注册限流中间件"""
        self.rate_limit_middleware = middleware
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if self.performance_middleware:
            return self.performance_middleware.get_metrics()
        return {}
    
    def reset_performance_metrics(self):
        """重置性能指标"""
        if self.performance_middleware:
            self.performance_middleware.request_count = 0
            self.performance_middleware.total_time = 0.0


# 全局中间件管理器实例
middleware_manager = MiddlewareManager()