#!/usr/bin/env python3
"""
API中间件

提供FastAPI应用的中间件，包括：
1. CORS中间件
2. 请求日志中间件
3. 错误处理中间件
4. 性能监控中间件
5. 安全中间件
6. 限流中间件
"""

import time
import uuid
import json
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from ..core.config import settings
from ..core.cache import CacheManager
from ..core.exceptions import (
    BaseAPIException,
    RateLimitException,
    ValidationException
)
from ..core.responses import ResponseBuilder
from ..utils.logger import get_logger


# =============================================================================
# 日志配置
# =============================================================================

logger = get_logger(__name__)
access_logger = get_logger("access")
performance_logger = get_logger("performance")
security_logger = get_logger("security")


# =============================================================================
# 请求日志中间件
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    
    记录所有HTTP请求的详细信息，包括：
    - 请求方法、路径、参数
    - 客户端IP、用户代理
    - 响应状态码、处理时间
    - 请求和响应大小
    """
    
    def __init__(self, app: ASGIApp, log_body: bool = False):
        """
        初始化请求日志中间件
        
        Args:
            app: ASGI应用
            log_body: 是否记录请求体
        """
        super().__init__(app)
        self.log_body = log_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个处理器
            
        Returns:
            响应对象
        """
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取客户端信息
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        
        # 记录请求信息
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "headers": dict(request.headers),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 记录请求体（如果启用）
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_info["body_size"] = len(body)
                    # 只记录JSON请求体
                    content_type = request.headers.get("content-type", "")
                    if "application/json" in content_type:
                        try:
                            request_info["body"] = json.loads(body.decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_info["body"] = "<binary_data>"
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")
        
        # 处理请求
        try:
            response = await call_next(request)
        except Exception as e:
            # 记录异常
            process_time = time.time() - start_time
            error_info = {
                **request_info,
                "status_code": 500,
                "process_time": process_time,
                "error": str(e),
                "error_type": type(e).__name__
            }
            access_logger.error(f"Request failed: {json.dumps(error_info)}")
            raise
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录响应信息
        response_info = {
            **request_info,
            "status_code": response.status_code,
            "process_time": process_time,
            "response_size": getattr(response, "content_length", 0)
        }
        
        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # 记录访问日志
        if response.status_code >= 400:
            access_logger.warning(f"Request completed: {json.dumps(response_info)}")
        else:
            access_logger.info(f"Request completed: {json.dumps(response_info)}")
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端IP地址
        
        Args:
            request: 请求对象
            
        Returns:
            客户端IP地址
        """
        # 检查代理头
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


# =============================================================================
# 性能监控中间件
# =============================================================================

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    性能监控中间件
    
    监控API性能指标，包括：
    - 响应时间统计
    - 慢查询检测
    - 内存使用监控
    - 并发请求统计
    """
    
    def __init__(self, app: ASGIApp, slow_request_threshold: float = 1.0):
        """
        初始化性能监控中间件
        
        Args:
            app: ASGI应用
            slow_request_threshold: 慢请求阈值（秒）
        """
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.active_requests = 0
        self.request_stats = {
            "total_requests": 0,
            "slow_requests": 0,
            "error_requests": 0,
            "avg_response_time": 0.0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个处理器
            
        Returns:
            响应对象
        """
        # 增加活跃请求计数
        self.active_requests += 1
        start_time = time.time()
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 更新统计信息
            self._update_stats(process_time, response.status_code)
            
            # 检查慢请求
            if process_time > self.slow_request_threshold:
                self._log_slow_request(request, process_time)
            
            # 添加性能头
            response.headers["X-Response-Time"] = f"{process_time:.3f}s"
            response.headers["X-Active-Requests"] = str(self.active_requests)
            
            return response
            
        except Exception as e:
            # 记录错误
            process_time = time.time() - start_time
            self._update_stats(process_time, 500, error=True)
            raise
        
        finally:
            # 减少活跃请求计数
            self.active_requests -= 1
    
    def _update_stats(self, process_time: float, status_code: int, error: bool = False):
        """
        更新统计信息
        
        Args:
            process_time: 处理时间
            status_code: 状态码
            error: 是否为错误
        """
        self.request_stats["total_requests"] += 1
        
        if error or status_code >= 400:
            self.request_stats["error_requests"] += 1
        
        if process_time > self.slow_request_threshold:
            self.request_stats["slow_requests"] += 1
        
        # 更新平均响应时间
        total = self.request_stats["total_requests"]
        current_avg = self.request_stats["avg_response_time"]
        self.request_stats["avg_response_time"] = (
            (current_avg * (total - 1) + process_time) / total
        )
    
    def _log_slow_request(self, request: Request, process_time: float):
        """
        记录慢请求
        
        Args:
            request: 请求对象
            process_time: 处理时间
        """
        slow_request_info = {
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "process_time": process_time,
            "threshold": self.slow_request_threshold,
            "query_params": dict(request.query_params),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        performance_logger.warning(
            f"Slow request detected: {json.dumps(slow_request_info)}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        return {
            **self.request_stats,
            "active_requests": self.active_requests,
            "slow_request_rate": (
                self.request_stats["slow_requests"] / 
                max(self.request_stats["total_requests"], 1)
            ),
            "error_rate": (
                self.request_stats["error_requests"] / 
                max(self.request_stats["total_requests"], 1)
            )
        }


# =============================================================================
# 安全中间件
# =============================================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    安全中间件
    
    提供基础安全功能，包括：
    - 安全头设置
    - 请求大小限制
    - 可疑请求检测
    - IP白名单/黑名单
    """
    
    def __init__(
        self,
        app: ASGIApp,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        blocked_ips: Optional[list] = None,
        allowed_ips: Optional[list] = None
    ):
        """
        初始化安全中间件
        
        Args:
            app: ASGI应用
            max_request_size: 最大请求大小
            blocked_ips: IP黑名单
            allowed_ips: IP白名单
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        self.blocked_ips = set(blocked_ips or [])
        self.allowed_ips = set(allowed_ips or [])
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个处理器
            
        Returns:
            响应对象
        """
        # 获取客户端IP
        client_ip = self._get_client_ip(request)
        
        # 检查IP黑名单
        if client_ip in self.blocked_ips:
            security_logger.warning(f"Blocked IP access attempt: {client_ip}")
            return JSONResponse(
                status_code=403,
                content=ResponseBuilder.error("访问被拒绝")
            )
        
        # 检查IP白名单（如果设置）
        if self.allowed_ips and client_ip not in self.allowed_ips:
            security_logger.warning(f"Non-whitelisted IP access attempt: {client_ip}")
            return JSONResponse(
                status_code=403,
                content=ResponseBuilder.error("访问被拒绝")
            )
        
        # 检查请求大小
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            security_logger.warning(
                f"Large request blocked: {content_length} bytes from {client_ip}"
            )
            return JSONResponse(
                status_code=413,
                content=ResponseBuilder.error("请求体过大")
            )
        
        # 检查可疑请求
        if self._is_suspicious_request(request):
            security_logger.warning(
                f"Suspicious request detected from {client_ip}: {request.url.path}"
            )
        
        # 处理请求
        response = await call_next(request)
        
        # 添加安全头
        self._add_security_headers(response)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端IP地址
        
        Args:
            request: 请求对象
            
        Returns:
            客户端IP地址
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _is_suspicious_request(self, request: Request) -> bool:
        """
        检查是否为可疑请求
        
        Args:
            request: 请求对象
            
        Returns:
            是否可疑
        """
        # 检查路径中的可疑模式
        suspicious_patterns = [
            "../", "..%2f", "..\\", "..%5c",
            "<script", "javascript:", "vbscript:",
            "union select", "drop table", "insert into",
            "/etc/passwd", "/proc/", "/sys/"
        ]
        
        path = request.url.path.lower()
        query = str(request.query_params).lower()
        
        for pattern in suspicious_patterns:
            if pattern in path or pattern in query:
                return True
        
        # 检查异常的User-Agent
        user_agent = request.headers.get("User-Agent", "").lower()
        suspicious_agents = [
            "sqlmap", "nmap", "nikto", "dirb", "gobuster",
            "burp", "owasp", "w3af", "acunetix"
        ]
        
        for agent in suspicious_agents:
            if agent in user_agent:
                return True
        
        return False
    
    def _add_security_headers(self, response: Response):
        """
        添加安全头
        
        Args:
            response: 响应对象
        """
        # 基础安全头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HSTS（仅HTTPS）
        if settings.ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # CSP（内容安全策略）
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none';"
        )


# =============================================================================
# 限流中间件
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    限流中间件
    
    基于IP地址的全局限流，防止API滥用
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst_limit: int = 10
    ):
        """
        初始化限流中间件
        
        Args:
            app: ASGI应用
            requests_per_minute: 每分钟请求限制
            burst_limit: 突发请求限制
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.cache = CacheManager()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求
        
        Args:
            request: 请求对象
            call_next: 下一个处理器
            
        Returns:
            响应对象
        """
        # 获取客户端IP
        client_ip = self._get_client_ip(request)
        
        # 检查限流
        if await self._is_rate_limited(client_ip):
            security_logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content=ResponseBuilder.error(
                    "请求过于频繁，请稍后重试",
                    error_code="RATE_LIMIT_EXCEEDED"
                ),
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # 处理请求
        response = await call_next(request)
        
        # 添加限流头
        remaining = await self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端IP地址
        
        Args:
            request: 请求对象
            
        Returns:
            客户端IP地址
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        """
        检查是否超出限流
        
        Args:
            client_ip: 客户端IP
            
        Returns:
            是否超出限流
        """
        # 分钟级限流键
        minute_key = f"rate_limit:minute:{client_ip}"
        # 突发限流键
        burst_key = f"rate_limit:burst:{client_ip}"
        
        # 检查分钟级限流
        minute_count = await self.cache.get(minute_key)
        if minute_count is None:
            await self.cache.set(minute_key, 1, expire=60)
            minute_count = 1
        else:
            minute_count = int(minute_count)
            if minute_count >= self.requests_per_minute:
                return True
            await self.cache.increment(minute_key)
        
        # 检查突发限流（10秒窗口）
        burst_count = await self.cache.get(burst_key)
        if burst_count is None:
            await self.cache.set(burst_key, 1, expire=10)
        else:
            burst_count = int(burst_count)
            if burst_count >= self.burst_limit:
                return True
            await self.cache.increment(burst_key)
        
        return False
    
    async def _get_remaining_requests(self, client_ip: str) -> int:
        """
        获取剩余请求次数
        
        Args:
            client_ip: 客户端IP
            
        Returns:
            剩余请求次数
        """
        minute_key = f"rate_limit:minute:{client_ip}"
        minute_count = await self.cache.get(minute_key)
        
        if minute_count is None:
            return self.requests_per_minute
        
        return max(0, self.requests_per_minute - int(minute_count))


# =============================================================================
# 中间件设置函数
# =============================================================================

def setup_api_middleware(app: FastAPI):
    """
    设置API中间件
    
    Args:
        app: FastAPI应用实例
    """
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time",
            "X-Response-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining"
        ]
    )
    
    # 信任主机中间件
    if settings.ALLOWED_HOSTS:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
    
    # Gzip压缩中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 安全中间件
    app.add_middleware(
        SecurityMiddleware,
        max_request_size=settings.MAX_REQUEST_SIZE,
        blocked_ips=getattr(settings, "BLOCKED_IPS", []),
        allowed_ips=getattr(settings, "ALLOWED_IPS", [])
    )
    
    # 限流中间件（生产环境）
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
            burst_limit=settings.RATE_LIMIT_BURST
        )
    
    # 性能监控中间件
    app.add_middleware(
        PerformanceMonitoringMiddleware,
        slow_request_threshold=settings.SLOW_REQUEST_THRESHOLD
    )
    
    # 请求日志中间件
    app.add_middleware(
        RequestLoggingMiddleware,
        log_body=settings.LOG_REQUEST_BODY
    )
    
    logger.info("API middleware setup completed")


def get_performance_stats(app: FastAPI) -> Dict[str, Any]:
    """
    获取性能统计信息
    
    Args:
        app: FastAPI应用实例
        
    Returns:
        性能统计字典
    """
    # 查找性能监控中间件
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, PerformanceMonitoringMiddleware):
            return middleware.cls.get_stats()
    
    return {"error": "Performance monitoring middleware not found"}