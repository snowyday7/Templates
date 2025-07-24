# -*- coding: utf-8 -*-
"""
限流中间件

提供API请求频率限制功能。
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

# 导入模板库组件
from src.core.cache import get_cache_client
from src.core.logging import get_logger
from src.utils.exceptions import RateLimitException

# 导入应用组件
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware:
    """限流中间件类"""
    
    def __init__(self):
        self.cache = get_cache_client()
        # 内存缓存作为备用（当Redis不可用时）
        self.memory_cache: Dict[str, Dict] = defaultdict(dict)
        self.cleanup_interval = 300  # 5分钟清理一次内存缓存
        self.last_cleanup = datetime.utcnow()
        
        # 默认限流配置
        self.default_limits = {
            "requests_per_minute": settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "requests_per_hour": settings.RATE_LIMIT_REQUESTS_PER_HOUR,
            "requests_per_day": settings.RATE_LIMIT_REQUESTS_PER_DAY
        }
        
        # 特殊路径的限流配置
        self.path_limits = {
            "/api/v1/auth/login": {
                "requests_per_minute": 5,
                "requests_per_hour": 20,
                "requests_per_day": 100
            },
            "/api/v1/auth/register": {
                "requests_per_minute": 3,
                "requests_per_hour": 10,
                "requests_per_day": 50
            },
            "/api/v1/messages/send": {
                "requests_per_minute": 20,
                "requests_per_hour": 200,
                "requests_per_day": 1000
            },
            "/api/v1/conversations": {
                "requests_per_minute": 30,
                "requests_per_hour": 300,
                "requests_per_day": 2000
            }
        }
    
    async def __call__(self, request: Request, call_next):
        """
        中间件处理函数
        """
        # 跳过不需要限流的路径
        if self._should_skip_rate_limit(request.url.path):
            return await call_next(request)
        
        try:
            # 获取客户端标识
            client_id = self._get_client_id(request)
            
            # 获取限流配置
            limits = self._get_rate_limits(request.url.path)
            
            # 检查限流
            await self._check_rate_limit(client_id, request.url.path, limits)
            
            # 记录请求
            await self._record_request(client_id, request.url.path)
            
            response = await call_next(request)
            
            # 添加限流头信息
            await self._add_rate_limit_headers(response, client_id, request.url.path, limits)
            
            return response
            
        except RateLimitException as e:
            logger.warning(f"限流触发: {e} (Client: {self._get_client_id(request)}, Path: {request.url.path})")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": str(e),
                    "retry_after": e.retry_after if hasattr(e, 'retry_after') else 60
                },
                headers={
                    "Retry-After": str(getattr(e, 'retry_after', 60)),
                    "X-RateLimit-Limit": str(getattr(e, 'limit', 0)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int((datetime.utcnow() + timedelta(seconds=getattr(e, 'retry_after', 60))).timestamp()))
                }
            )
        except Exception as e:
            logger.error(f"限流中间件错误: {e}")
            # 限流错误不应该阻止请求处理
            return await call_next(request)
    
    def _should_skip_rate_limit(self, path: str) -> bool:
        """
        判断是否跳过限流
        """
        # 不需要限流的路径
        skip_paths = [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
        
        # 检查精确匹配
        if path in skip_paths:
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
    
    def _get_client_id(self, request: Request) -> str:
        """
        获取客户端标识
        """
        # 优先使用用户ID（如果已认证）
        if hasattr(request.state, "user_id") and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        # 使用IP地址
        client_ip = request.client.host
        
        # 检查是否有代理IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    def _get_rate_limits(self, path: str) -> Dict[str, int]:
        """
        获取路径的限流配置
        """
        # 检查是否有特殊配置
        for pattern, limits in self.path_limits.items():
            if path.startswith(pattern) or path == pattern:
                return limits
        
        # 返回默认配置
        return self.default_limits
    
    async def _check_rate_limit(self, client_id: str, path: str, limits: Dict[str, int]):
        """
        检查限流
        """
        now = datetime.utcnow()
        
        # 检查每分钟限制
        if "requests_per_minute" in limits:
            await self._check_time_window(
                client_id, path, "minute", 
                limits["requests_per_minute"], 
                60, now
            )
        
        # 检查每小时限制
        if "requests_per_hour" in limits:
            await self._check_time_window(
                client_id, path, "hour", 
                limits["requests_per_hour"], 
                3600, now
            )
        
        # 检查每日限制
        if "requests_per_day" in limits:
            await self._check_time_window(
                client_id, path, "day", 
                limits["requests_per_day"], 
                86400, now
            )
    
    async def _check_time_window(self, client_id: str, path: str, window: str, limit: int, seconds: int, now: datetime):
        """
        检查时间窗口内的请求数量
        """
        key = f"rate_limit:{client_id}:{path}:{window}"
        
        try:
            # 尝试使用Redis
            if self.cache:
                count = await self._check_redis_rate_limit(key, limit, seconds, now)
            else:
                count = await self._check_memory_rate_limit(key, limit, seconds, now)
            
            if count > limit:
                raise RateLimitException(
                    f"超过{window}请求限制 ({count}/{limit})",
                    limit=limit,
                    retry_after=seconds
                )
                
        except RateLimitException:
            raise
        except Exception as e:
            logger.error(f"检查限流失败: {e}")
            # 限流检查失败时，允许请求通过
    
    async def _check_redis_rate_limit(self, key: str, limit: int, seconds: int, now: datetime) -> int:
        """
        使用Redis检查限流
        """
        try:
            # 使用滑动窗口算法
            pipe = self.cache.pipeline()
            
            # 移除过期的记录
            cutoff = now - timedelta(seconds=seconds)
            pipe.zremrangebyscore(key, 0, cutoff.timestamp())
            
            # 添加当前请求
            pipe.zadd(key, {str(now.timestamp()): now.timestamp()})
            
            # 获取当前计数
            pipe.zcard(key)
            
            # 设置过期时间
            pipe.expire(key, seconds)
            
            results = await pipe.execute()
            count = results[2]  # zcard的结果
            
            return count
            
        except Exception as e:
            logger.error(f"Redis限流检查失败: {e}")
            # 回退到内存缓存
            return await self._check_memory_rate_limit(key, limit, seconds, now)
    
    async def _check_memory_rate_limit(self, key: str, limit: int, seconds: int, now: datetime) -> int:
        """
        使用内存缓存检查限流
        """
        # 清理过期缓存
        await self._cleanup_memory_cache()
        
        if key not in self.memory_cache:
            self.memory_cache[key] = {"requests": [], "expires": now + timedelta(seconds=seconds)}
        
        cache_entry = self.memory_cache[key]
        
        # 移除过期的请求
        cutoff = now - timedelta(seconds=seconds)
        cache_entry["requests"] = [
            req_time for req_time in cache_entry["requests"] 
            if req_time > cutoff
        ]
        
        # 添加当前请求
        cache_entry["requests"].append(now)
        cache_entry["expires"] = now + timedelta(seconds=seconds)
        
        return len(cache_entry["requests"])
    
    async def _record_request(self, client_id: str, path: str):
        """
        记录请求（用于统计）
        """
        try:
            if self.cache:
                # 记录到Redis
                stats_key = f"rate_limit_stats:{client_id}:{datetime.utcnow().strftime('%Y-%m-%d')}"
                await self.cache.hincrby(stats_key, path, 1)
                await self.cache.expire(stats_key, 86400 * 7)  # 保留7天
        except Exception as e:
            logger.warning(f"记录请求统计失败: {e}")
    
    async def _add_rate_limit_headers(self, response, client_id: str, path: str, limits: Dict[str, int]):
        """
        添加限流头信息
        """
        try:
            # 获取当前使用量
            if "requests_per_minute" in limits:
                key = f"rate_limit:{client_id}:{path}:minute"
                if self.cache:
                    remaining = limits["requests_per_minute"] - await self.cache.zcard(key)
                else:
                    cache_entry = self.memory_cache.get(key, {"requests": []})
                    remaining = limits["requests_per_minute"] - len(cache_entry["requests"])
                
                response.headers["X-RateLimit-Limit"] = str(limits["requests_per_minute"])
                response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
                response.headers["X-RateLimit-Reset"] = str(int((datetime.utcnow() + timedelta(minutes=1)).timestamp()))
                
        except Exception as e:
            logger.warning(f"添加限流头信息失败: {e}")
    
    async def _cleanup_memory_cache(self):
        """
        清理过期的内存缓存
        """
        now = datetime.utcnow()
        
        # 每5分钟清理一次
        if (now - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return
        
        self.last_cleanup = now
        
        # 移除过期的缓存项
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry.get("expires", now) < now
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        logger.debug(f"清理了 {len(expired_keys)} 个过期的限流缓存项")


class CustomRateLimitMiddleware(RateLimitMiddleware):
    """自定义限流中间件"""
    
    def __init__(self, custom_limits: Optional[Dict[str, Dict[str, int]]] = None):
        super().__init__()
        if custom_limits:
            self.path_limits.update(custom_limits)
    
    def add_path_limit(self, path: str, limits: Dict[str, int]):
        """
        添加路径限流配置
        """
        self.path_limits[path] = limits
    
    def remove_path_limit(self, path: str):
        """
        移除路径限流配置
        """
        if path in self.path_limits:
            del self.path_limits[path]
    
    def update_default_limits(self, limits: Dict[str, int]):
        """
        更新默认限流配置
        """
        self.default_limits.update(limits)


class IPWhitelistMiddleware:
    """IP白名单中间件（跳过限流）"""
    
    def __init__(self, whitelist: Optional[list] = None):
        self.whitelist = whitelist or []
        # 添加本地IP
        self.whitelist.extend(["127.0.0.1", "localhost", "::1"])
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取客户端IP
        """
        client_ip = request.client.host
        
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return client_ip
    
    async def __call__(self, request: Request, call_next):
        """
        中间件处理函数
        """
        client_ip = self._get_client_ip(request)
        
        # 如果IP在白名单中，跳过后续的限流中间件
        if client_ip in self.whitelist:
            request.state.skip_rate_limit = True
        
        return await call_next(request)


def create_rate_limit_middleware(config: Optional[dict] = None) -> RateLimitMiddleware:
    """
    创建限流中间件实例
    """
    if config:
        return CustomRateLimitMiddleware(config.get("custom_limits"))
    return RateLimitMiddleware()