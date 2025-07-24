# -*- coding: utf-8 -*-
"""
缓存模块

提供Redis和内存缓存功能。
"""

import sys
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
import json
import pickle
from datetime import datetime, timedelta
from functools import lru_cache

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# 导入模板库组件
from src.core.cache import BaseCache
from src.core.logging import get_logger

# 导入应用组件
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class MemoryCache:
    """
    内存缓存实现
    """
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        """
        if key not in self._cache:
            return None
        
        item = self._cache[key]
        
        # 检查是否过期
        if item.get("expires_at") and datetime.utcnow() > item["expires_at"]:
            del self._cache[key]
            return None
        
        return item["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值
        """
        # 如果缓存已满，删除最旧的项
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
            del self._cache[oldest_key]
        
        expires_at = None
        if ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        self._cache[key] = {
            "value": value,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at
        }
        
        return True
    
    def delete(self, key: str) -> bool:
        """
        删除缓存值
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        """
        return self.get(key) is not None
    
    def clear(self) -> bool:
        """
        清空缓存
        """
        self._cache.clear()
        return True
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        获取匹配模式的键列表
        """
        import fnmatch
        return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    def size(self) -> int:
        """
        获取缓存大小
        """
        return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """
        清理过期的缓存项
        """
        now = datetime.utcnow()
        expired_keys = []
        
        for key, item in self._cache.items():
            if item.get("expires_at") and now > item["expires_at"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)


class RedisCache:
    """
    Redis缓存实现
    """
    
    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        """
        try:
            value = self._redis.get(key)
            if value is None:
                return None
            
            # 尝试反序列化
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                try:
                    return pickle.loads(value)
                except (pickle.PickleError, TypeError):
                    return value.decode() if isinstance(value, bytes) else value
        
        except RedisError as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值
        """
        try:
            # 序列化值
            if isinstance(value, (dict, list, tuple)):
                serialized_value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (int, float, bool)):
                serialized_value = json.dumps(value)
            elif isinstance(value, str):
                serialized_value = value
            else:
                serialized_value = pickle.dumps(value)
            
            if ttl:
                return self._redis.setex(key, ttl, serialized_value)
            else:
                return self._redis.set(key, serialized_value)
        
        except RedisError as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存值
        """
        try:
            return bool(self._redis.delete(key))
        except RedisError as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        """
        try:
            return bool(self._redis.exists(key))
        except RedisError as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def clear(self) -> bool:
        """
        清空缓存
        """
        try:
            return self._redis.flushdb()
        except RedisError as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        获取匹配模式的键列表
        """
        try:
            keys = self._redis.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except RedisError as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    def ttl(self, key: str) -> int:
        """
        获取键的TTL
        """
        try:
            return self._redis.ttl(key)
        except RedisError as e:
            logger.error(f"Redis ttl error: {e}")
            return -1
    
    def expire(self, key: str, ttl: int) -> bool:
        """
        设置键的过期时间
        """
        try:
            return bool(self._redis.expire(key, ttl))
        except RedisError as e:
            logger.error(f"Redis expire error: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        递增计数器
        """
        try:
            return self._redis.incrby(key, amount)
        except RedisError as e:
            logger.error(f"Redis increment error: {e}")
            return None
    
    def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """
        递减计数器
        """
        try:
            return self._redis.decrby(key, amount)
        except RedisError as e:
            logger.error(f"Redis decrement error: {e}")
            return None


class Cache(BaseCache):
    """
    缓存管理类
    """
    
    def __init__(self):
        super().__init__()
        self._redis_client: Optional[redis.Redis] = None
        self._redis_cache: Optional[RedisCache] = None
        self._memory_cache: MemoryCache = MemoryCache()
        self._use_redis = False
        
        self._initialize()
    
    def _initialize(self) -> None:
        """
        初始化缓存
        """
        if REDIS_AVAILABLE and settings.REDIS_URL:
            try:
                self._redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # 测试连接
                self._redis_client.ping()
                self._redis_cache = RedisCache(self._redis_client)
                self._use_redis = True
                
                logger.info("Redis cache initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}. Falling back to memory cache.")
                self._use_redis = False
        else:
            logger.info("Using memory cache (Redis not available or not configured)")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        """
        if self._use_redis and self._redis_cache:
            return self._redis_cache.get(key)
        else:
            return self._memory_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值
        """
        if self._use_redis and self._redis_cache:
            return self._redis_cache.set(key, value, ttl)
        else:
            return self._memory_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """
        删除缓存值
        """
        if self._use_redis and self._redis_cache:
            return self._redis_cache.delete(key)
        else:
            return self._memory_cache.delete(key)
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        """
        if self._use_redis and self._redis_cache:
            return self._redis_cache.exists(key)
        else:
            return self._memory_cache.exists(key)
    
    def clear(self) -> bool:
        """
        清空缓存
        """
        if self._use_redis and self._redis_cache:
            return self._redis_cache.clear()
        else:
            return self._memory_cache.clear()
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        获取匹配模式的键列表
        """
        if self._use_redis and self._redis_cache:
            return self._redis_cache.keys(pattern)
        else:
            return self._memory_cache.keys(pattern)
    
    def health_check(self) -> bool:
        """
        健康检查
        """
        try:
            if self._use_redis and self._redis_client:
                self._redis_client.ping()
                return True
            else:
                # 内存缓存总是可用的
                return True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        """
        info = {
            "type": "redis" if self._use_redis else "memory",
            "available": self.health_check()
        }
        
        if self._use_redis and self._redis_client:
            try:
                redis_info = self._redis_client.info()
                info.update({
                    "redis_version": redis_info.get("redis_version"),
                    "used_memory": redis_info.get("used_memory_human"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "total_commands_processed": redis_info.get("total_commands_processed")
                })
            except Exception as e:
                logger.error(f"Failed to get Redis info: {e}")
        else:
            info.update({
                "memory_cache_size": self._memory_cache.size(),
                "max_size": self._memory_cache._max_size
            })
        
        return info
    
    # 便捷方法
    def get_or_set(self, key: str, func, ttl: Optional[int] = None) -> Any:
        """
        获取缓存值，如果不存在则调用函数设置
        """
        value = self.get(key)
        if value is None:
            value = func()
            self.set(key, value, ttl)
        return value
    
    def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """
        递增计数器
        """
        if self._use_redis and self._redis_cache:
            result = self._redis_cache.increment(key, amount)
            if result is not None and ttl:
                self._redis_cache.expire(key, ttl)
            return result or 0
        else:
            current = self.get(key) or 0
            new_value = current + amount
            self.set(key, new_value, ttl)
            return new_value
    
    def decrement(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """
        递减计数器
        """
        if self._use_redis and self._redis_cache:
            result = self._redis_cache.decrement(key, amount)
            if result is not None and ttl:
                self._redis_cache.expire(key, ttl)
            return result or 0
        else:
            current = self.get(key) or 0
            new_value = max(0, current - amount)
            self.set(key, new_value, ttl)
            return new_value
    
    def cleanup(self) -> int:
        """
        清理过期缓存
        """
        if not self._use_redis:
            return self._memory_cache.cleanup_expired()
        return 0


# 全局缓存实例
_cache: Optional[Cache] = None


@lru_cache()
def get_cache() -> Cache:
    """
    获取缓存实例（单例模式）
    """
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


# 便捷函数
def cache_get(key: str) -> Optional[Any]:
    """
    获取缓存值
    """
    return get_cache().get(key)


def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """
    设置缓存值
    """
    return get_cache().set(key, value, ttl)


def cache_delete(key: str) -> bool:
    """
    删除缓存值
    """
    return get_cache().delete(key)


def cache_exists(key: str) -> bool:
    """
    检查缓存键是否存在
    """
    return get_cache().exists(key)


def cache_clear() -> bool:
    """
    清空缓存
    """
    return get_cache().clear()


def cache_get_or_set(key: str, func, ttl: Optional[int] = None) -> Any:
    """
    获取缓存值，如果不存在则调用函数设置
    """
    return get_cache().get_or_set(key, func, ttl)


def cache_increment(key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
    """
    递增计数器
    """
    return get_cache().increment(key, amount, ttl)


def cache_decrement(key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
    """
    递减计数器
    """
    return get_cache().decrement(key, amount, ttl)