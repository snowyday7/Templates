# -*- coding: utf-8 -*-
"""
性能优化模块
提供缓存、异步处理、数据库优化、内存管理等性能提升功能
"""

import asyncio
import aioredis
import time
import functools
import threading
import multiprocessing
from typing import (
    Dict, List, Any, Optional, Union, Callable, Tuple,
    AsyncIterator, Iterator, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pickle
import hashlib
import weakref
import gc
import psutil
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import heapq
from collections import defaultdict, OrderedDict, deque
import sys
import tracemalloc
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
import redis
from cachetools import TTLCache, LRUCache, LFUCache
import asyncpg
import aiomysql
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheType(Enum):
    """缓存类型枚举"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    FILE = "file"
    DATABASE = "database"


class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最不经常使用
    TTL = "ttl"  # 生存时间
    FIFO = "fifo"  # 先进先出
    LIFO = "lifo"  # 后进先出


class OptimizationType(Enum):
    """优化类型枚举"""
    CACHE = "cache"
    ASYNC = "async"
    DATABASE = "database"
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"


@dataclass
class CacheConfig:
    """缓存配置"""
    cache_type: CacheType
    strategy: CacheStrategy = CacheStrategy.LRU
    max_size: int = 1000
    ttl: int = 3600  # 秒
    redis_url: Optional[str] = None
    redis_db: int = 0
    file_path: Optional[str] = None
    compression: bool = False
    serializer: str = "json"  # json, pickle
    key_prefix: str = ""
    namespace: str = "default"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    concurrent_requests: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheInterface(ABC):
    """缓存接口"""
    
    @abstractmethod
    async def get(self, key: str) -> Any:
        """获取缓存值"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取键列表"""
        pass


class MemoryCache(CacheInterface):
    """内存缓存"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
        
        if config.strategy == CacheStrategy.LRU:
            self.cache = LRUCache(maxsize=config.max_size)
        elif config.strategy == CacheStrategy.LFU:
            self.cache = LFUCache(maxsize=config.max_size)
        elif config.strategy == CacheStrategy.TTL:
            self.cache = TTLCache(maxsize=config.max_size, ttl=config.ttl)
        else:
            self.cache = {}
        
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Any:
        async with self._lock:
            full_key = f"{self.config.key_prefix}{key}"
            try:
                value = self.cache[full_key]
                self.stats["hits"] += 1
                return value
            except KeyError:
                self.stats["misses"] += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self._lock:
            full_key = f"{self.config.key_prefix}{key}"
            
            if self.config.strategy == CacheStrategy.TTL and ttl:
                # 为TTL缓存设置过期时间
                self.cache[full_key] = value
            else:
                self.cache[full_key] = value
            
            self.stats["sets"] += 1
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            full_key = f"{self.config.key_prefix}{key}"
            try:
                del self.cache[full_key]
                self.stats["deletes"] += 1
                return True
            except KeyError:
                return False
    
    async def exists(self, key: str) -> bool:
        async with self._lock:
            full_key = f"{self.config.key_prefix}{key}"
            return full_key in self.cache
    
    async def clear(self) -> bool:
        async with self._lock:
            self.cache.clear()
            return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        async with self._lock:
            if pattern == "*":
                return list(self.cache.keys())
            else:
                # 简单的模式匹配
                import fnmatch
                return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.config.max_size
        }


class RedisCache(CacheInterface):
    """Redis缓存"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
    
    async def connect(self):
        """连接Redis"""
        if not self.redis:
            self.redis = await aioredis.from_url(
                self.config.redis_url or "redis://localhost:6379",
                db=self.config.redis_db,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def disconnect(self):
        """断开Redis连接"""
        if self.redis:
            await self.redis.close()
    
    def _serialize(self, value: Any) -> str:
        """序列化值"""
        if self.config.serializer == "json":
            return json.dumps(value, ensure_ascii=False)
        elif self.config.serializer == "pickle":
            import base64
            return base64.b64encode(pickle.dumps(value)).decode()
        else:
            return str(value)
    
    def _deserialize(self, value: str) -> Any:
        """反序列化值"""
        if self.config.serializer == "json":
            return json.loads(value)
        elif self.config.serializer == "pickle":
            import base64
            return pickle.loads(base64.b64decode(value.encode()))
        else:
            return value
    
    async def get(self, key: str) -> Any:
        await self.connect()
        full_key = f"{self.config.namespace}:{self.config.key_prefix}{key}"
        
        try:
            value = await self.redis.get(full_key)
            if value is not None:
                self.stats["hits"] += 1
                return self._deserialize(value)
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        await self.connect()
        full_key = f"{self.config.namespace}:{self.config.key_prefix}{key}"
        
        try:
            serialized_value = self._serialize(value)
            expire_time = ttl or self.config.ttl
            
            await self.redis.setex(full_key, expire_time, serialized_value)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        await self.connect()
        full_key = f"{self.config.namespace}:{self.config.key_prefix}{key}"
        
        try:
            result = await self.redis.delete(full_key)
            if result > 0:
                self.stats["deletes"] += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        await self.connect()
        full_key = f"{self.config.namespace}:{self.config.key_prefix}{key}"
        
        try:
            return await self.redis.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        await self.connect()
        pattern = f"{self.config.namespace}:{self.config.key_prefix}*"
        
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        await self.connect()
        full_pattern = f"{self.config.namespace}:{self.config.key_prefix}{pattern}"
        
        try:
            keys = await self.redis.keys(full_pattern)
            # 移除命名空间和前缀
            prefix_len = len(f"{self.config.namespace}:{self.config.key_prefix}")
            return [key[prefix_len:] for key in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.caches: Dict[str, CacheInterface] = {}
        self.default_cache: Optional[str] = None
    
    def register_cache(self, name: str, cache: CacheInterface, is_default: bool = False):
        """注册缓存"""
        self.caches[name] = cache
        if is_default or not self.default_cache:
            self.default_cache = name
    
    def get_cache(self, name: Optional[str] = None) -> CacheInterface:
        """获取缓存实例"""
        cache_name = name or self.default_cache
        if not cache_name or cache_name not in self.caches:
            raise ValueError(f"Cache '{cache_name}' not found")
        return self.caches[cache_name]
    
    async def get(self, key: str, cache_name: Optional[str] = None) -> Any:
        """获取缓存值"""
        cache = self.get_cache(cache_name)
        return await cache.get(key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_name: Optional[str] = None
    ) -> bool:
        """设置缓存值"""
        cache = self.get_cache(cache_name)
        return await cache.set(key, value, ttl)
    
    async def delete(self, key: str, cache_name: Optional[str] = None) -> bool:
        """删除缓存值"""
        cache = self.get_cache(cache_name)
        return await cache.delete(key)
    
    async def clear_all(self):
        """清空所有缓存"""
        for cache in self.caches.values():
            await cache.clear()


def cache_decorator(
    ttl: int = 3600,
    cache_name: Optional[str] = None,
    key_func: Optional[Callable] = None
):
    """缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            cache_manager = CacheManager()
            cached_value = await cache_manager.get(cache_key, cache_name)
            
            if cached_value is not None:
                return cached_value
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl, cache_name)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同步版本的缓存装饰器
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 简化的同步缓存实现
            if not hasattr(sync_wrapper, '_cache'):
                sync_wrapper._cache = {}
            
            if cache_key in sync_wrapper._cache:
                cached_time, cached_value = sync_wrapper._cache[cache_key]
                if time.time() - cached_time < ttl:
                    return cached_value
            
            result = func(*args, **kwargs)
            sync_wrapper._cache[cache_key] = (time.time(), result)
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AsyncTaskManager:
    """异步任务管理器"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.task_queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_stats = defaultdict(int)
    
    async def submit_task(
        self,
        task_id: str,
        coro_or_func: Union[Callable, Callable[..., Any]],
        *args,
        use_process: bool = False,
        **kwargs
    ) -> str:
        """提交异步任务"""
        if task_id in self.running_tasks:
            raise ValueError(f"Task {task_id} is already running")
        
        if asyncio.iscoroutinefunction(coro_or_func):
            # 协程函数
            task = asyncio.create_task(coro_or_func(*args, **kwargs))
        elif use_process:
            # 使用进程池
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(self.process_pool, coro_or_func, *args, **kwargs)
        else:
            # 使用线程池
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(self.thread_pool, coro_or_func, *args, **kwargs)
        
        self.running_tasks[task_id] = task
        self.task_stats["submitted"] += 1
        
        # 添加完成回调
        task.add_done_callback(lambda t: self._task_completed(task_id, t))
        
        return task_id
    
    def _task_completed(self, task_id: str, task: asyncio.Task):
        """任务完成回调"""
        try:
            result = task.result()
            self.task_results[task_id] = result
            self.task_stats["completed"] += 1
        except Exception as e:
            self.task_results[task_id] = {"error": str(e)}
            self.task_stats["failed"] += 1
        finally:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        if task_id in self.running_tasks:
            try:
                result = await asyncio.wait_for(self.running_tasks[task_id], timeout=timeout)
                return result
            except asyncio.TimeoutError:
                raise TimeoutError(f"Task {task_id} timed out")
        
        raise ValueError(f"Task {task_id} not found")
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self.running_tasks[task_id]
            self.task_stats["cancelled"] += 1
            return True
        
        return False
    
    async def wait_for_all(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """等待所有任务完成"""
        if not self.running_tasks:
            return self.task_results
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.running_tasks.values(), return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not complete within timeout")
        
        return self.task_results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取任务统计"""
        return {
            "running": len(self.running_tasks),
            "completed": self.task_stats["completed"],
            "failed": self.task_stats["failed"],
            "cancelled": self.task_stats["cancelled"],
            "submitted": self.task_stats["submitted"]
        }
    
    async def cleanup(self):
        """清理资源"""
        # 取消所有运行中的任务
        for task_id in list(self.running_tasks.keys()):
            await self.cancel_task(task_id)
        
        # 关闭线程池和进程池
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class DatabaseOptimizer:
    """数据库优化器"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool = None
        self.query_cache = LRUCache(maxsize=1000)
        self.query_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})
    
    async def create_connection_pool(self, **kwargs):
        """创建连接池"""
        if "postgresql" in self.database_url:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                **kwargs
            )
        elif "mysql" in self.database_url:
            # MySQL连接池配置
            import urllib.parse as urlparse
            parsed = urlparse.urlparse(self.database_url)
            
            self.connection_pool = await aiomysql.create_pool(
                host=parsed.hostname,
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                db=parsed.path[1:],  # 移除开头的'/'
                minsize=5,
                maxsize=20,
                **kwargs
            )
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        use_cache: bool = True,
        cache_ttl: int = 300
    ) -> List[Dict[str, Any]]:
        """执行查询"""
        start_time = time.time()
        
        # 生成缓存键
        cache_key = hashlib.md5(f"{query}{params}".encode()).hexdigest()
        
        # 尝试从缓存获取
        if use_cache and cache_key in self.query_cache:
            cached_time, cached_result = self.query_cache[cache_key]
            if time.time() - cached_time < cache_ttl:
                return cached_result
        
        # 执行查询
        try:
            if "postgresql" in self.database_url:
                result = await self._execute_postgresql_query(query, params)
            elif "mysql" in self.database_url:
                result = await self._execute_mysql_query(query, params)
            else:
                raise ValueError("Unsupported database type")
            
            # 缓存结果
            if use_cache:
                self.query_cache[cache_key] = (time.time(), result)
            
            # 更新统计
            execution_time = time.time() - start_time
            self.query_stats[query]["count"] += 1
            self.query_stats[query]["total_time"] += execution_time
            
            return result
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def _execute_postgresql_query(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """执行PostgreSQL查询"""
        async with self.connection_pool.acquire() as conn:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)
            
            return [dict(row) for row in rows]
    
    async def _execute_mysql_query(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """执行MySQL查询"""
        async with self.connection_pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query, params)
                rows = await cursor.fetchall()
                return rows
    
    async def execute_batch(
        self,
        queries: List[Tuple[str, Optional[Tuple]]],
        use_transaction: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """批量执行查询"""
        results = []
        
        if "postgresql" in self.database_url:
            async with self.connection_pool.acquire() as conn:
                if use_transaction:
                    async with conn.transaction():
                        for query, params in queries:
                            if params:
                                rows = await conn.fetch(query, *params)
                            else:
                                rows = await conn.fetch(query)
                            results.append([dict(row) for row in rows])
                else:
                    for query, params in queries:
                        if params:
                            rows = await conn.fetch(query, *params)
                        else:
                            rows = await conn.fetch(query)
                        results.append([dict(row) for row in rows])
        
        return results
    
    def get_query_stats(self) -> Dict[str, Any]:
        """获取查询统计"""
        stats = {}
        for query, data in self.query_stats.items():
            avg_time = data["total_time"] / data["count"] if data["count"] > 0 else 0
            stats[query[:100]] = {  # 截断长查询
                "count": data["count"],
                "total_time": data["total_time"],
                "avg_time": avg_time
            }
        
        return stats
    
    async def optimize_table(self, table_name: str):
        """优化表"""
        if "postgresql" in self.database_url:
            await self.execute_query(f"VACUUM ANALYZE {table_name}")
        elif "mysql" in self.database_url:
            await self.execute_query(f"OPTIMIZE TABLE {table_name}")
    
    async def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """获取表统计信息"""
        if "postgresql" in self.database_url:
            query = """
            SELECT 
                schemaname,
                tablename,
                attname,
                n_distinct,
                correlation
            FROM pg_stats 
            WHERE tablename = $1
            """
            return await self.execute_query(query, (table_name,))
        elif "mysql" in self.database_url:
            query = "SHOW TABLE STATUS LIKE %s"
            return await self.execute_query(query, (table_name,))
        
        return {}
    
    async def close(self):
        """关闭连接池"""
        if self.connection_pool:
            if "postgresql" in self.database_url:
                await self.connection_pool.close()
            elif "mysql" in self.database_url:
                self.connection_pool.close()
                await self.connection_pool.wait_closed()


class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        self.snapshots = []
        self.is_tracing = False
    
    def start_tracing(self):
        """开始内存追踪"""
        if not self.is_tracing:
            tracemalloc.start()
            self.is_tracing = True
    
    def stop_tracing(self):
        """停止内存追踪"""
        if self.is_tracing:
            tracemalloc.stop()
            self.is_tracing = False
    
    def take_snapshot(self, description: str = "") -> int:
        """拍摄内存快照"""
        if not self.is_tracing:
            self.start_tracing()
        
        snapshot = tracemalloc.take_snapshot()
        snapshot_id = len(self.snapshots)
        
        self.snapshots.append({
            "id": snapshot_id,
            "timestamp": datetime.now(),
            "description": description,
            "snapshot": snapshot
        })
        
        return snapshot_id
    
    def compare_snapshots(
        self,
        snapshot1_id: int,
        snapshot2_id: int,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """比较内存快照"""
        if snapshot1_id >= len(self.snapshots) or snapshot2_id >= len(self.snapshots):
            raise ValueError("Invalid snapshot ID")
        
        snapshot1 = self.snapshots[snapshot1_id]["snapshot"]
        snapshot2 = self.snapshots[snapshot2_id]["snapshot"]
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        result = {
            "snapshot1": {
                "id": snapshot1_id,
                "timestamp": self.snapshots[snapshot1_id]["timestamp"],
                "description": self.snapshots[snapshot1_id]["description"]
            },
            "snapshot2": {
                "id": snapshot2_id,
                "timestamp": self.snapshots[snapshot2_id]["timestamp"],
                "description": self.snapshots[snapshot2_id]["description"]
            },
            "top_differences": []
        }
        
        for stat in top_stats[:top_n]:
            result["top_differences"].append({
                "file": stat.traceback.format()[0] if stat.traceback else "Unknown",
                "size_diff": stat.size_diff,
                "count_diff": stat.count_diff,
                "size": stat.size,
                "count": stat.count
            })
        
        return result
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # 物理内存
            "vms": memory_info.vms,  # 虚拟内存
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "total": psutil.virtual_memory().total
        }
    
    def get_top_memory_consumers(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """获取内存消耗最大的对象"""
        if not self.is_tracing:
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        result = []
        for stat in top_stats[:top_n]:
            result.append({
                "file": stat.traceback.format()[0] if stat.traceback else "Unknown",
                "size": stat.size,
                "count": stat.count,
                "size_mb": stat.size / 1024 / 1024
            })
        
        return result


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.memory_profiler = MemoryProfiler()
    
    async def start_monitoring(self):
        """开始性能监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.memory_profiler.start_tracing()
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """停止性能监控"""
        if self.is_monitoring:
            self.is_monitoring = False
            self.memory_profiler.stop_tracing()
            
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 保持历史记录在合理范围内
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                await asyncio.sleep(self.sample_interval)
            
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.sample_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        process = psutil.Process()
        
        # CPU使用率
        cpu_usage = process.cpu_percent()
        
        # 内存使用情况
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / 1024 / 1024  # MB
        
        # 网络IO
        try:
            net_io = psutil.net_io_counters()
            network_bytes = net_io.bytes_sent + net_io.bytes_recv
        except:
            network_bytes = 0
        
        # 磁盘IO
        try:
            disk_io = psutil.disk_io_counters()
            disk_bytes = disk_io.read_bytes + disk_io.write_bytes
        except:
            disk_bytes = 0
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            metadata={
                "network_bytes": network_bytes,
                "disk_bytes": disk_bytes,
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
        )
    
    def get_metrics_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}
        
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        cpu_values = [m.cpu_usage for m in metrics]
        memory_values = [m.memory_usage for m in metrics]
        
        return {
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "current": memory_values[-1] if memory_values else 0
            },
            "sample_count": len(metrics),
            "time_range": {
                "start": metrics[0].timestamp if metrics else None,
                "end": metrics[-1].timestamp if metrics else None
            }
        }
    
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """检测性能问题"""
        issues = []
        
        if not self.metrics_history:
            return issues
        
        recent_metrics = self.metrics_history[-10:]  # 最近10个样本
        
        # 检查高CPU使用率
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > 80:
            issues.append({
                "type": "high_cpu_usage",
                "severity": "high" if avg_cpu > 90 else "medium",
                "description": f"平均CPU使用率过高: {avg_cpu:.1f}%",
                "value": avg_cpu
            })
        
        # 检查高内存使用率
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        total_memory = psutil.virtual_memory().total / 1024 / 1024  # MB
        memory_percent = (avg_memory / total_memory) * 100
        
        if memory_percent > 80:
            issues.append({
                "type": "high_memory_usage",
                "severity": "high" if memory_percent > 90 else "medium",
                "description": f"平均内存使用率过高: {memory_percent:.1f}%",
                "value": memory_percent
            })
        
        # 检查内存泄漏（内存持续增长）
        if len(recent_metrics) >= 5:
            memory_trend = []
            for i in range(1, len(recent_metrics)):
                memory_trend.append(recent_metrics[i].memory_usage - recent_metrics[i-1].memory_usage)
            
            avg_growth = sum(memory_trend) / len(memory_trend)
            if avg_growth > 10:  # 每次采样增长超过10MB
                issues.append({
                    "type": "memory_leak",
                    "severity": "high",
                    "description": f"检测到可能的内存泄漏，平均增长: {avg_growth:.1f}MB/sample",
                    "value": avg_growth
                })
        
        return issues


# 性能优化工具函数
def profile_function(func: Callable) -> Callable:
    """函数性能分析装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(
                f"Function {func.__name__} executed in {execution_time:.4f}s, "
                f"memory delta: {memory_delta / 1024 / 1024:.2f}MB"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            raise
    
    return wrapper


async def batch_process(
    items: List[T],
    process_func: Callable[[T], Any],
    batch_size: int = 100,
    max_concurrency: int = 10
) -> List[Any]:
    """批量处理数据"""
    results = []
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_item(item):
        async with semaphore:
            if asyncio.iscoroutinefunction(process_func):
                return await process_func(item)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, process_func, item)
    
    # 分批处理
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = [process_item(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)
    
    return results


# 使用示例
if __name__ == "__main__":
    async def example_usage():
        # 缓存管理
        cache_manager = CacheManager()
        
        # 注册内存缓存
        memory_config = CacheConfig(
            cache_type=CacheType.MEMORY,
            strategy=CacheStrategy.LRU,
            max_size=1000
        )
        memory_cache = MemoryCache(memory_config)
        cache_manager.register_cache("memory", memory_cache, is_default=True)
        
        # 使用缓存
        await cache_manager.set("test_key", "test_value")
        value = await cache_manager.get("test_key")
        print(f"Cached value: {value}")
        
        # 异步任务管理
        task_manager = AsyncTaskManager()
        
        async def sample_task(x):
            await asyncio.sleep(1)
            return x * 2
        
        task_id = await task_manager.submit_task("task1", sample_task, 5)
        result = await task_manager.get_task_result(task_id)
        print(f"Task result: {result}")
        
        # 性能监控
        monitor = PerformanceMonitor()
        await monitor.start_monitoring()
        
        # 模拟一些工作
        await asyncio.sleep(5)
        
        summary = monitor.get_metrics_summary()
        print(f"Performance summary: {summary}")
        
        issues = monitor.detect_performance_issues()
        if issues:
            print(f"Performance issues detected: {issues}")
        
        await monitor.stop_monitoring()
        await task_manager.cleanup()
    
    # 运行示例
    asyncio.run(example_usage())