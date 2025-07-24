# -*- coding: utf-8 -*-
"""
请求日志中间件

提供请求和响应的日志记录功能。
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# 导入模板库组件
from src.core.logging import get_logger
from src.core.cache import get_cache_client

# 导入应用组件
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件类"""
    
    def __init__(self, app, log_level: str = "INFO", include_request_body: bool = False, include_response_body: bool = False, max_body_size: int = 1024):
        super().__init__(app)
        self.log_level = log_level.upper()
        self.include_request_body = include_request_body
        self.include_response_body = include_response_body
        self.max_body_size = max_body_size
        self.cache = get_cache_client()
        
        # 不记录日志的路径
        self.skip_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]
        
        # 不记录请求体的路径（敏感信息）
        self.skip_body_paths = [
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/reset-password",
            "/api/v1/users/password"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """
        中间件处理函数
        """
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 跳过不需要记录的路径
        if self._should_skip_logging(request.url.path):
            return await call_next(request)
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 记录请求信息
        await self._log_request(request, request_id)
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录响应信息
            await self._log_response(request, response, request_id, process_time)
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response
            
        except Exception as e:
            # 记录异常
            process_time = time.time() - start_time
            await self._log_exception(request, e, request_id, process_time)
            raise
    
    def _should_skip_logging(self, path: str) -> bool:
        """
        判断是否跳过日志记录
        """
        # 检查精确匹配
        if path in self.skip_paths:
            return True
        
        # 检查前缀匹配
        skip_prefixes = [
            "/static/",
            "/docs",
            "/redoc"
        ]
        
        for prefix in skip_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def _should_skip_body_logging(self, path: str) -> bool:
        """
        判断是否跳过请求体日志记录
        """
        return any(path.startswith(skip_path) for skip_path in self.skip_body_paths)
    
    async def _log_request(self, request: Request, request_id: str):
        """
        记录请求信息
        """
        try:
            # 获取客户端信息
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")
            
            # 获取用户信息
            user_id = getattr(request.state, "user_id", None)
            
            # 构建请求日志
            log_data = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "user_id": user_id,
                "content_type": request.headers.get("Content-Type"),
                "content_length": request.headers.get("Content-Length")
            }
            
            # 记录请求体（如果启用且不在跳过列表中）
            if (self.include_request_body and 
                not self._should_skip_body_logging(request.url.path) and
                request.method in ["POST", "PUT", "PATCH"]):
                
                try:
                    body = await self._get_request_body(request)
                    if body:
                        log_data["request_body"] = body
                except Exception as e:
                    log_data["request_body_error"] = str(e)
            
            # 记录到日志
            logger.info(f"Request started: {request.method} {request.url.path}", extra={
                "request_data": log_data,
                "request_id": request_id
            })
            
            # 缓存请求信息（用于后续分析）
            if self.cache:
                await self._cache_request_info(request_id, log_data)
                
        except Exception as e:
            logger.error(f"记录请求日志失败: {e}")
    
    async def _log_response(self, request: Request, response: Response, request_id: str, process_time: float):
        """
        记录响应信息
        """
        try:
            # 构建响应日志
            log_data = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "process_time": process_time,
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length")
            }
            
            # 记录响应体（如果启用）
            if self.include_response_body:
                try:
                    body = await self._get_response_body(response)
                    if body:
                        log_data["response_body"] = body
                except Exception as e:
                    log_data["response_body_error"] = str(e)
            
            # 确定日志级别
            if response.status_code >= 500:
                log_level = "ERROR"
            elif response.status_code >= 400:
                log_level = "WARNING"
            else:
                log_level = "INFO"
            
            # 记录到日志
            getattr(logger, log_level.lower())(
                f"Request completed: {request.method} {request.url.path} - {response.status_code} ({process_time:.4f}s)",
                extra={
                    "response_data": log_data,
                    "request_id": request_id
                }
            )
            
            # 更新缓存信息
            if self.cache:
                await self._update_cached_request_info(request_id, log_data)
                
        except Exception as e:
            logger.error(f"记录响应日志失败: {e}")
    
    async def _log_exception(self, request: Request, exception: Exception, request_id: str, process_time: float):
        """
        记录异常信息
        """
        try:
            log_data = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "process_time": process_time
            }
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {type(exception).__name__}: {exception} ({process_time:.4f}s)",
                extra={
                    "exception_data": log_data,
                    "request_id": request_id
                },
                exc_info=True
            )
            
            # 更新缓存信息
            if self.cache:
                await self._update_cached_request_info(request_id, log_data)
                
        except Exception as e:
            logger.error(f"记录异常日志失败: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端IP地址
        """
        # 检查代理头
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # 返回直接连接的IP
        return request.client.host if request.client else "unknown"
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """
        获取请求体内容
        """
        try:
            body = await request.body()
            if not body:
                return None
            
            # 限制大小
            if len(body) > self.max_body_size:
                return f"[Body too large: {len(body)} bytes, truncated to {self.max_body_size}]" + body[:self.max_body_size].decode("utf-8", errors="ignore")
            
            # 尝试解析为JSON
            try:
                return json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 如果不是JSON，返回原始字符串
                return body.decode("utf-8", errors="ignore")
                
        except Exception:
            return None
    
    async def _get_response_body(self, response: Response) -> Optional[str]:
        """
        获取响应体内容
        """
        try:
            # 注意：这里需要小心处理，因为响应体可能已经被消费
            # 在实际应用中，可能需要使用流式处理或其他方法
            if hasattr(response, 'body'):
                body = response.body
                if body and len(body) <= self.max_body_size:
                    try:
                        return json.loads(body.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return body.decode("utf-8", errors="ignore")
            return None
        except Exception:
            return None
    
    async def _cache_request_info(self, request_id: str, log_data: Dict[str, Any]):
        """
        缓存请求信息
        """
        try:
            cache_key = f"request_log:{request_id}"
            await self.cache.setex(
                cache_key,
                3600,  # 1小时过期
                json.dumps(log_data, default=str)
            )
        except Exception as e:
            logger.warning(f"缓存请求信息失败: {e}")
    
    async def _update_cached_request_info(self, request_id: str, additional_data: Dict[str, Any]):
        """
        更新缓存的请求信息
        """
        try:
            cache_key = f"request_log:{request_id}"
            cached_data = await self.cache.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                data.update(additional_data)
                await self.cache.setex(
                    cache_key,
                    3600,
                    json.dumps(data, default=str)
                )
        except Exception as e:
            logger.warning(f"更新缓存请求信息失败: {e}")


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """性能日志中间件"""
    
    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # 记录慢请求
        if process_time > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} took {process_time:.4f}s",
                extra={
                    "slow_request": {
                        "method": request.method,
                        "path": request.url.path,
                        "process_time": process_time,
                        "threshold": self.slow_request_threshold,
                        "user_id": getattr(request.state, "user_id", None)
                    }
                }
            )
        
        return response


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """安全日志中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # 可疑的用户代理
        self.suspicious_user_agents = [
            "sqlmap",
            "nikto",
            "nmap",
            "masscan",
            "nessus",
            "openvas",
            "w3af",
            "skipfish"
        ]
        
        # 可疑的路径模式
        self.suspicious_paths = [
            "/.env",
            "/admin",
            "/wp-admin",
            "/phpmyadmin",
            "/.git",
            "/config",
            "/backup",
            "/.aws",
            "/.ssh"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # 检查可疑活动
        await self._check_suspicious_activity(request)
        
        response = await call_next(request)
        
        # 检查响应中的安全问题
        await self._check_response_security(request, response)
        
        return response
    
    async def _check_suspicious_activity(self, request: Request):
        """
        检查可疑活动
        """
        user_agent = request.headers.get("User-Agent", "").lower()
        path = request.url.path.lower()
        
        # 检查可疑用户代理
        for suspicious_ua in self.suspicious_user_agents:
            if suspicious_ua in user_agent:
                logger.warning(
                    f"Suspicious user agent detected: {user_agent}",
                    extra={
                        "security_event": {
                            "type": "suspicious_user_agent",
                            "user_agent": user_agent,
                            "path": request.url.path,
                            "client_ip": self._get_client_ip(request),
                            "method": request.method
                        }
                    }
                )
                break
        
        # 检查可疑路径
        for suspicious_path in self.suspicious_paths:
            if suspicious_path in path:
                logger.warning(
                    f"Suspicious path access: {request.url.path}",
                    extra={
                        "security_event": {
                            "type": "suspicious_path",
                            "path": request.url.path,
                            "client_ip": self._get_client_ip(request),
                            "user_agent": user_agent,
                            "method": request.method
                        }
                    }
                )
                break
    
    async def _check_response_security(self, request: Request, response: Response):
        """
        检查响应安全性
        """
        # 检查是否返回了敏感信息
        if response.status_code == 500:
            logger.error(
                f"Internal server error: {request.method} {request.url.path}",
                extra={
                    "security_event": {
                        "type": "internal_server_error",
                        "path": request.url.path,
                        "client_ip": self._get_client_ip(request),
                        "user_id": getattr(request.state, "user_id", None)
                    }
                }
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端IP地址
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


def create_logging_middleware(
    include_request_body: bool = False,
    include_response_body: bool = False,
    max_body_size: int = 1024,
    slow_request_threshold: float = 1.0
) -> list:
    """
    创建日志中间件列表
    """
    return [
        SecurityLoggingMiddleware,
        PerformanceLoggingMiddleware,
        lambda app: RequestLoggingMiddleware(
            app,
            include_request_body=include_request_body,
            include_response_body=include_response_body,
            max_body_size=max_body_size
        )
    ]