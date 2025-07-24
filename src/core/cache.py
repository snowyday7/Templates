#!/usr/bin/env python3
"""
Redis缓存管理模块

提供统一的缓存接口，支持：
1. Redis连接管理
2. 缓存操作（get, set, delete等）
3. 分布式锁
4. 发布订阅
5. 缓存装饰器
6. 缓存策略
"""

import asyncio
import json
import logging
import pickle
from contextlib import asynccontextmanager
from functools import wraps
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    AsyncGenerator,
    TypeVar,
    Generic,
)
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.asyncio.sentinel import Sentinel
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config import get_settings, get_redis_url


# 配置日志
logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')


class CacheSerializer:
    """
    缓存序列化器
    
    支持多种序列化方式
    """
    
    @staticmethod
    def serialize_json(data: Any) -> str:
        """JSON序列化"""
        return json.dumps(data, ensure_ascii=False, default=str)
    
    @staticmethod
    def deserialize_json(data: str) -> Any:
        """JSON反序列化"""
        return json.loads(data)
    
    @staticmethod
    def serialize_pickle(data: Any) -> bytes:
        """Pickle序列化"""
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize_pickle(data: bytes) -> Any:
        """Pickle反序列化"""
        return pickle.loads(data)


class DistributedLock:
    """
    分布式锁
    
    基于Redis实现的分布式锁
    """
    
    def __init__(
        self,
        redis_client: Redis,
        key: str,
        timeout: int = 10,
        blocking_timeout: Optional[int] = None,
    ):
        self.redis_client = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.blocking_timeout = blocking_timeout
        self.identifier = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.release()
    
    async def acquire(self) -> bool:
        """
        获取锁
        
        Returns:
            bool: 是否成功获取锁
        """
        import uuid
        
        self.identifier = str(uuid.uuid4())
        end_time = None
        
        if self.blocking_timeout is not None:
            end_time = asyncio.get_event_loop().time() + self.blocking_timeout
        
        while True:
            # 尝试获取锁
            if await self.redis_client.set(
                self.key,
                self.identifier,
                nx=True,
                ex=self.timeout,
            ):
                logger.debug(f"Acquired lock: {self.key}")
                return True
            
            # 检查是否超时
            if end_time and asyncio.get_event_loop().time() > end_time:
                logger.warning(f"Failed to acquire lock: {self.key} (timeout)")
                return False
            
            # 等待一段时间后重试
            await asyncio.sleep(0.001)
    
    async def release(self) -> bool:
        """
        释放锁
        
        Returns:
            bool: 是否成功释放锁
        """
        if not self.identifier:
            return False
        
        # Lua脚本确保原子性
        lua_script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        
        try:
            result = await self.redis_client.eval(
                lua_script,
                1,
                self.key,
                self.identifier,
            )
            
            if result:
                logger.debug(f"Released lock: {self.key}")
                return True
            else:
                logger.warning(f"Failed to release lock: {self.key} (not owner)")
                return False
                
        except Exception as e:
            logger.error(f"Error releasing lock {self.key}: {e}")
            return False
        finally:
            self.identifier = None


class CacheManager:
    """
    缓存管理器
    
    负责管理Redis连接和缓存操作
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._redis: Optional[Redis] = None
        self._sentinel: Optional[Sentinel] = None
        self._pool: Optional[ConnectionPool] = None
        self._initialized = False
        self.serializer = CacheSerializer()
    
    async def initialize(self) -> None:
        """
        初始化Redis连接
        """
        if self._initialized:
            return
        
        try:
            # 检查是否使用Sentinel
            if self.settings.redis_sentinel_hosts:
                await self._initialize_sentinel()
            else:
                await self._initialize_direct()
            
            # 测试连接
            await self._test_connection()
            
            self._initialized = True
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    async def _initialize_sentinel(self) -> None:
        """
        初始化Sentinel连接
        """
        sentinel_hosts = [
            tuple(host.split(":"))
            for host in self.settings.redis_sentinel_hosts.split(",")
        ]
        
        self._sentinel = Sentinel(
            sentinel_hosts,
            password=self.settings.redis_sentinel_password,
            socket_timeout=self.settings.redis_socket_timeout,
            socket_connect_timeout=self.settings.redis_socket_connect_timeout,
        )
        
        self._redis = self._sentinel.master_for(
            self.settings.redis_sentinel_service,
            password=self.settings.redis_password,
            db=self.settings.redis_db,
            socket_timeout=self.settings.redis_socket_timeout,
            socket_connect_timeout=self.settings.redis_socket_connect_timeout,
            health_check_interval=self.settings.redis_health_check_interval,
        )
    
    async def _initialize_direct(self) -> None:
        """
        初始化直连Redis
        """
        redis_url = get_redis_url(self.settings)
        
        self._pool = ConnectionPool.from_url(
            redis_url,
            max_connections=self.settings.redis_max_connections,
            socket_timeout=self.settings.redis_socket_timeout,
            socket_connect_timeout=self.settings.redis_socket_connect_timeout,
            health_check_interval=self.settings.redis_health_check_interval,
            retry_on_timeout=True,
        )
        
        self._redis = Redis(connection_pool=self._pool)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RedisError, ConnectionError, TimeoutError)),
    )
    async def _test_connection(self) -> None:
        """
        测试Redis连接
        """
        try:
            await self._redis.ping()
            logger.info("Redis connection test successful")
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            raise
    
    async def close(self) -> None:
        """
        关闭Redis连接
        """
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
        
        self._initialized = False
        logger.info("Redis cache connections closed")
    
    def _make_key(self, key: str) -> str:
        """
        生成缓存键
        
        Args:
            key: 原始键
        
        Returns:
            str: 完整的缓存键
        """
        return f"{self.settings.cache_key_prefix}{self.settings.cache_version}:{key}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
        
        Returns:
            Any: 缓存值
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(key)
            value = await self._redis.get(cache_key)
            
            if value is None:
                return default
            
            # 尝试JSON反序列化
            try:
                return self.serializer.deserialize_json(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 如果JSON反序列化失败，尝试Pickle
                try:
                    return self.serializer.deserialize_pickle(value)
                except Exception:
                    # 如果都失败，返回原始值
                    return value.decode('utf-8') if isinstance(value, bytes) else value
                    
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        timeout: Optional[int] = None,
        serialize_method: str = "json",
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            timeout: 过期时间（秒）
            serialize_method: 序列化方法（json或pickle）
        
        Returns:
            bool: 是否成功设置
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(key)
            
            # 序列化值
            if serialize_method == "json":
                serialized_value = self.serializer.serialize_json(value)
            elif serialize_method == "pickle":
                serialized_value = self.serializer.serialize_pickle(value)
            else:
                serialized_value = str(value)
            
            # 设置过期时间
            if timeout is None:
                timeout = self.settings.cache_default_timeout
            
            await self._redis.set(cache_key, serialized_value, ex=timeout)
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
        
        Returns:
            bool: 是否成功删除
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(key)
            result = await self._redis.delete(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        检查缓存键是否存在
        
        Args:
            key: 缓存键
        
        Returns:
            bool: 是否存在
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(key)
            result = await self._redis.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error checking cache key existence {key}: {e}")
            return False
    
    async def expire(self, key: str, timeout: int) -> bool:
        """
        设置缓存键过期时间
        
        Args:
            key: 缓存键
            timeout: 过期时间（秒）
        
        Returns:
            bool: 是否成功设置
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(key)
            result = await self._redis.expire(cache_key, timeout)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting cache key expiration {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """
        获取缓存键剩余生存时间
        
        Args:
            key: 缓存键
        
        Returns:
            int: 剩余生存时间（秒），-1表示永不过期，-2表示不存在
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(key)
            return await self._redis.ttl(cache_key)
            
        except Exception as e:
            logger.error(f"Error getting cache key TTL {key}: {e}")
            return -2
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """
        递增缓存值
        
        Args:
            key: 缓存键
            amount: 递增量
        
        Returns:
            int: 递增后的值
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(key)
            return await self._redis.incrby(cache_key, amount)
            
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return 0
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        """
        递减缓存值
        
        Args:
            key: 缓存键
            amount: 递减量
        
        Returns:
            int: 递减后的值
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_key = self._make_key(key)
            return await self._redis.decrby(cache_key, amount)
            
        except Exception as e:
            logger.error(f"Error decrementing cache key {key}: {e}")
            return 0
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        批量获取缓存值
        
        Args:
            keys: 缓存键列表
        
        Returns:
            Dict[str, Any]: 缓存值字典
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_keys = [self._make_key(key) for key in keys]
            values = await self._redis.mget(cache_keys)
            
            result = {}
            for i, key in enumerate(keys):
                value = values[i]
                if value is not None:
                    try:
                        result[key] = self.serializer.deserialize_json(value.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        try:
                            result[key] = self.serializer.deserialize_pickle(value)
                        except Exception:
                            result[key] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting multiple cache keys: {e}")
            return {}
    
    async def set_many(
        self,
        mapping: Dict[str, Any],
        timeout: Optional[int] = None,
        serialize_method: str = "json",
    ) -> bool:
        """
        批量设置缓存值
        
        Args:
            mapping: 键值对字典
            timeout: 过期时间（秒）
            serialize_method: 序列化方法
        
        Returns:
            bool: 是否成功设置
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            pipe = self._redis.pipeline()
            
            for key, value in mapping.items():
                cache_key = self._make_key(key)
                
                # 序列化值
                if serialize_method == "json":
                    serialized_value = self.serializer.serialize_json(value)
                elif serialize_method == "pickle":
                    serialized_value = self.serializer.serialize_pickle(value)
                else:
                    serialized_value = str(value)
                
                # 设置过期时间
                if timeout is None:
                    timeout = self.settings.cache_default_timeout
                
                pipe.set(cache_key, serialized_value, ex=timeout)
            
            await pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Error setting multiple cache keys: {e}")
            return False
    
    async def delete_many(self, keys: List[str]) -> int:
        """
        批量删除缓存值
        
        Args:
            keys: 缓存键列表
        
        Returns:
            int: 删除的键数量
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_keys = [self._make_key(key) for key in keys]
            return await self._redis.delete(*cache_keys)
            
        except Exception as e:
            logger.error(f"Error deleting multiple cache keys: {e}")
            return 0
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        根据模式清除缓存
        
        Args:
            pattern: 匹配模式
        
        Returns:
            int: 删除的键数量
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cache_pattern = self._make_key(pattern)
            keys = await self._redis.keys(cache_pattern)
            
            if keys:
                return await self._redis.delete(*keys)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0
    
    async def flush_all(self) -> bool:
        """
        清空所有缓存
        
        Returns:
            bool: 是否成功清空
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            await self._redis.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Error flushing all cache: {e}")
            return False
    
    def get_lock(self, key: str, timeout: int = 10, blocking_timeout: Optional[int] = None) -> DistributedLock:
        """
        获取分布式锁
        
        Args:
            key: 锁键
            timeout: 锁超时时间
            blocking_timeout: 阻塞超时时间
        
        Returns:
            DistributedLock: 分布式锁实例
        """
        return DistributedLock(self._redis, key, timeout, blocking_timeout)
    
    async def publish(self, channel: str, message: Any) -> int:
        """
        发布消息
        
        Args:
            channel: 频道
            message: 消息
        
        Returns:
            int: 接收消息的订阅者数量
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            serialized_message = self.serializer.serialize_json(message)
            return await self._redis.publish(channel, serialized_message)
            
        except Exception as e:
            logger.error(f"Error publishing message to channel {channel}: {e}")
            return 0
    
    async def subscribe(self, *channels: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        订阅频道
        
        Args:
            channels: 频道列表
        
        Yields:
            Dict[str, Any]: 消息
        """
        if not self._initialized:
            await self.initialize()
        
        pubsub = self._redis.pubsub()
        
        try:
            await pubsub.subscribe(*channels)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = self.serializer.deserialize_json(message['data'].decode('utf-8'))
                        yield {
                            'channel': message['channel'].decode('utf-8'),
                            'data': data,
                            'type': message['type'],
                        }
                    except Exception as e:
                        logger.error(f"Error deserializing message: {e}")
                        
        except Exception as e:
            logger.error(f"Error in subscription: {e}")
        finally:
            await pubsub.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Redis健康检查
        
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            start_time = asyncio.get_event_loop().time()
            await self._redis.ping()
            response_time = asyncio.get_event_loop().time() - start_time
            
            # 获取Redis信息
            info = await self._redis.info()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "timestamp": asyncio.get_event_loop().time(),
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time(),
            }


# 全局缓存管理器实例
cache_manager = CacheManager()


# 缓存装饰器
def cached(
    key_pattern: str,
    timeout: Optional[int] = None,
    serialize_method: str = "json",
    key_builder: Optional[Callable] = None,
):
    """
    缓存装饰器
    
    Args:
        key_pattern: 缓存键模式
        timeout: 过期时间
        serialize_method: 序列化方法
        key_builder: 自定义键构建器
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 构建缓存键
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # 默认键构建逻辑
                func_name = func.__name__
                args_str = "_".join(str(arg) for arg in args)
                kwargs_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
                cache_key = key_pattern.format(
                    func=func_name,
                    args=args_str,
                    kwargs=kwargs_str,
                )
            
            # 尝试从缓存获取
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存储到缓存
            await cache_manager.set(
                cache_key,
                result,
                timeout=timeout,
                serialize_method=serialize_method,
            )
            
            logger.debug(f"Cache miss for key: {cache_key}, result cached")
            return result
            
        return wrapper
    return decorator


# 缓存初始化函数
async def init_cache() -> None:
    """
    初始化缓存
    """
    await cache_manager.initialize()


# 缓存关闭函数
async def close_cache() -> None:
    """
    关闭缓存连接
    """
    await cache_manager.close()


# 缓存健康检查函数
async def cache_health_check() -> Dict[str, Any]:
    """
    缓存健康检查
    
    Returns:
        Dict[str, Any]: 健康检查结果
    """
    return await cache_manager.health_check()