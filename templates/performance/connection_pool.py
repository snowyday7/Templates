"""连接池管理系统

提供各种连接池的统一管理，包括：
- 数据库连接池
- Redis连接池
- HTTP连接池
- 连接池监控和优化
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Optional, List, Union, Callable, AsyncGenerator, Generator
from dataclasses import dataclass, field
from enum import Enum

import aioredis
import asyncpg
import httpx
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PoolStatus(str, Enum):
    """连接池状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class PoolStats:
    """连接池统计信息"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    pending_requests: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    peak_connections: int = 0
    last_reset: float = field(default_factory=time.time)
    
    @property
    def utilization_rate(self) -> float:
        """连接池利用率"""
        if self.total_connections == 0:
            return 0.0
        return self.active_connections / self.total_connections * 100
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 100.0
        return (self.total_requests - self.failed_requests) / self.total_requests * 100


class PoolConfig(BaseSettings):
    """连接池配置"""
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 60
    
    class Config:
        env_prefix = "POOL_"


class BaseConnectionPool(ABC):
    """连接池基类"""
    
    def __init__(self, config: PoolConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.stats = PoolStats()
        self.status = PoolStatus.OFFLINE
        self._lock = threading.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._callbacks: Dict[str, List[Callable]] = {
            'on_connect': [],
            'on_disconnect': [],
            'on_error': [],
            'on_health_check': []
        }
    
    @abstractmethod
    async def initialize(self):
        """初始化连接池"""
        pass
    
    @abstractmethod
    async def close(self):
        """关闭连接池"""
        pass
    
    @abstractmethod
    async def get_connection(self):
        """获取连接"""
        pass
    
    @abstractmethod
    async def return_connection(self, connection):
        """归还连接"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """移除回调函数"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    async def _trigger_callbacks(self, event: str, *args, **kwargs):
        """触发回调函数"""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error in {event}: {e}")
    
    async def start_health_check(self):
        """启动健康检查"""
        if self._health_check_task:
            return
        
        async def health_check_loop():
            while True:
                try:
                    is_healthy = await self.health_check()
                    old_status = self.status
                    
                    if is_healthy:
                        self.status = PoolStatus.HEALTHY
                    else:
                        self.status = PoolStatus.CRITICAL
                    
                    if old_status != self.status:
                        await self._trigger_callbacks('on_health_check', self.status)
                    
                    await asyncio.sleep(self.config.health_check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Health check error: {e}")
                    await asyncio.sleep(self.config.health_check_interval)
        
        self._health_check_task = asyncio.create_task(health_check_loop())
    
    async def stop_health_check(self):
        """停止健康检查"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
    
    def get_stats(self) -> PoolStats:
        """获取统计信息"""
        return self.stats
    
    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self.stats = PoolStats()


class DatabaseConnectionPool(BaseConnectionPool):
    """数据库连接池"""
    
    def __init__(self, database_url: str, config: PoolConfig, name: str = "database"):
        super().__init__(config, name)
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        self._async_pool = None
    
    async def initialize(self):
        """初始化数据库连接池"""
        try:
            # 同步连接池
            self.engine = create_engine(
                self.database_url,
                poolclass=pool.QueuePool,
                pool_size=self.config.min_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=False
            )
            
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            
            # 异步连接池（如果是PostgreSQL）
            if "postgresql" in self.database_url:
                self._async_pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=self.config.min_size,
                    max_size=self.config.max_size,
                    command_timeout=self.config.pool_timeout
                )
            
            self.stats.total_connections = self.config.min_size
            self.status = PoolStatus.HEALTHY
            await self._trigger_callbacks('on_connect')
            
        except Exception as e:
            self.status = PoolStatus.OFFLINE
            await self._trigger_callbacks('on_error', e)
            raise
    
    async def close(self):
        """关闭连接池"""
        await self.stop_health_check()
        
        if self._async_pool:
            await self._async_pool.close()
        
        if self.engine:
            self.engine.dispose()
        
        self.status = PoolStatus.OFFLINE
        await self._trigger_callbacks('on_disconnect')
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取同步数据库会话"""
        if not self.session_factory:
            raise RuntimeError("Database pool not initialized")
        
        session = self.session_factory()
        start_time = time.time()
        
        try:
            with self._lock:
                self.stats.active_connections += 1
                self.stats.total_requests += 1
            
            yield session
            session.commit()
            
        except Exception as e:
            session.rollback()
            with self._lock:
                self.stats.failed_requests += 1
            raise
        finally:
            session.close()
            
            with self._lock:
                self.stats.active_connections -= 1
                response_time = time.time() - start_time
                self.stats.average_response_time = (
                    (self.stats.average_response_time * (self.stats.total_requests - 1) + response_time)
                    / self.stats.total_requests
                )
    
    async def get_connection(self):
        """获取异步数据库连接"""
        if not self._async_pool:
            raise RuntimeError("Async pool not available")
        
        return await self._async_pool.acquire()
    
    async def return_connection(self, connection):
        """归还异步连接"""
        if self._async_pool:
            await self._async_pool.release(connection)
    
    @asynccontextmanager
    async def get_async_connection(self):
        """获取异步连接上下文管理器"""
        connection = await self.get_connection()
        start_time = time.time()
        
        try:
            with self._lock:
                self.stats.active_connections += 1
                self.stats.total_requests += 1
            
            yield connection
            
        except Exception as e:
            with self._lock:
                self.stats.failed_requests += 1
            raise
        finally:
            await self.return_connection(connection)
            
            with self._lock:
                self.stats.active_connections -= 1
                response_time = time.time() - start_time
                self.stats.average_response_time = (
                    (self.stats.average_response_time * (self.stats.total_requests - 1) + response_time)
                    / self.stats.total_requests
                )
    
    async def health_check(self) -> bool:
        """数据库健康检查"""
        try:
            if self._async_pool:
                async with self.get_async_connection() as conn:
                    await conn.fetchval("SELECT 1")
            else:
                with self.get_session() as session:
                    session.execute("SELECT 1")
            return True
        except Exception:
            return False


class RedisConnectionPool(BaseConnectionPool):
    """Redis连接池"""
    
    def __init__(self, redis_url: str, config: PoolConfig, name: str = "redis"):
        super().__init__(config, name)
        self.redis_url = redis_url
        self.pool = None
    
    async def initialize(self):
        """初始化Redis连接池"""
        try:
            self.pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.config.max_size,
                retry_on_timeout=True,
                health_check_interval=self.config.health_check_interval
            )
            
            self.stats.total_connections = self.config.max_size
            self.status = PoolStatus.HEALTHY
            await self._trigger_callbacks('on_connect')
            
        except Exception as e:
            self.status = PoolStatus.OFFLINE
            await self._trigger_callbacks('on_error', e)
            raise
    
    async def close(self):
        """关闭Redis连接池"""
        await self.stop_health_check()
        
        if self.pool:
            await self.pool.disconnect()
        
        self.status = PoolStatus.OFFLINE
        await self._trigger_callbacks('on_disconnect')
    
    async def get_connection(self):
        """获取Redis连接"""
        if not self.pool:
            raise RuntimeError("Redis pool not initialized")
        
        return aioredis.Redis(connection_pool=self.pool)
    
    async def return_connection(self, connection):
        """归还Redis连接（Redis连接会自动管理）"""
        if connection:
            await connection.close()
    
    @asynccontextmanager
    async def get_redis(self) -> AsyncGenerator[aioredis.Redis, None]:
        """获取Redis连接上下文管理器"""
        redis = await self.get_connection()
        start_time = time.time()
        
        try:
            with self._lock:
                self.stats.active_connections += 1
                self.stats.total_requests += 1
            
            yield redis
            
        except Exception as e:
            with self._lock:
                self.stats.failed_requests += 1
            raise
        finally:
            await self.return_connection(redis)
            
            with self._lock:
                self.stats.active_connections -= 1
                response_time = time.time() - start_time
                self.stats.average_response_time = (
                    (self.stats.average_response_time * (self.stats.total_requests - 1) + response_time)
                    / self.stats.total_requests
                )
    
    async def health_check(self) -> bool:
        """Redis健康检查"""
        try:
            async with self.get_redis() as redis:
                await redis.ping()
            return True
        except Exception:
            return False


class HTTPConnectionPool(BaseConnectionPool):
    """HTTP连接池"""
    
    def __init__(self, config: PoolConfig, name: str = "http"):
        super().__init__(config, name)
        self.client = None
    
    async def initialize(self):
        """初始化HTTP连接池"""
        try:
            limits = httpx.Limits(
                max_keepalive_connections=self.config.max_size,
                max_connections=self.config.max_size + self.config.max_overflow,
                keepalive_expiry=self.config.pool_recycle
            )
            
            timeout = httpx.Timeout(
                connect=self.config.pool_timeout,
                read=self.config.pool_timeout,
                write=self.config.pool_timeout,
                pool=self.config.pool_timeout
            )
            
            self.client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                verify=True
            )
            
            self.stats.total_connections = self.config.max_size
            self.status = PoolStatus.HEALTHY
            await self._trigger_callbacks('on_connect')
            
        except Exception as e:
            self.status = PoolStatus.OFFLINE
            await self._trigger_callbacks('on_error', e)
            raise
    
    async def close(self):
        """关闭HTTP连接池"""
        await self.stop_health_check()
        
        if self.client:
            await self.client.aclose()
        
        self.status = PoolStatus.OFFLINE
        await self._trigger_callbacks('on_disconnect')
    
    async def get_connection(self):
        """获取HTTP客户端"""
        if not self.client:
            raise RuntimeError("HTTP pool not initialized")
        
        return self.client
    
    async def return_connection(self, connection):
        """归还HTTP连接（HTTP客户端会自动管理）"""
        pass
    
    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """发送HTTP请求"""
        start_time = time.time()
        
        try:
            with self._lock:
                self.stats.active_connections += 1
                self.stats.total_requests += 1
            
            response = await self.client.request(method, url, **kwargs)
            return response
            
        except Exception as e:
            with self._lock:
                self.stats.failed_requests += 1
            raise
        finally:
            with self._lock:
                self.stats.active_connections -= 1
                response_time = time.time() - start_time
                self.stats.average_response_time = (
                    (self.stats.average_response_time * (self.stats.total_requests - 1) + response_time)
                    / self.stats.total_requests
                )
    
    async def health_check(self) -> bool:
        """HTTP健康检查"""
        try:
            # 这里可以配置一个健康检查URL
            return self.client is not None
        except Exception:
            return False


class ConnectionPoolManager:
    """连接池管理器"""
    
    def __init__(self):
        self.pools: Dict[str, BaseConnectionPool] = {}
        self._lock = threading.Lock()
    
    def register_pool(self, name: str, pool: BaseConnectionPool):
        """注册连接池"""
        with self._lock:
            self.pools[name] = pool
    
    def get_pool(self, name: str) -> Optional[BaseConnectionPool]:
        """获取连接池"""
        return self.pools.get(name)
    
    async def initialize_all(self):
        """初始化所有连接池"""
        for pool in self.pools.values():
            await pool.initialize()
            await pool.start_health_check()
    
    async def close_all(self):
        """关闭所有连接池"""
        for pool in self.pools.values():
            await pool.close()
    
    def get_all_stats(self) -> Dict[str, PoolStats]:
        """获取所有连接池统计信息"""
        return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    def get_health_status(self) -> Dict[str, PoolStatus]:
        """获取所有连接池健康状态"""
        return {name: pool.status for name, pool in self.pools.items()}


# 全局连接池管理器
pool_manager = ConnectionPoolManager()


# 便捷函数
def get_db_pool(name: str = "default") -> Optional[DatabaseConnectionPool]:
    """获取数据库连接池"""
    pool = pool_manager.get_pool(f"db_{name}")
    return pool if isinstance(pool, DatabaseConnectionPool) else None


def get_redis_pool(name: str = "default") -> Optional[RedisConnectionPool]:
    """获取Redis连接池"""
    pool = pool_manager.get_pool(f"redis_{name}")
    return pool if isinstance(pool, RedisConnectionPool) else None


def get_http_pool(name: str = "default") -> Optional[HTTPConnectionPool]:
    """获取HTTP连接池"""
    pool = pool_manager.get_pool(f"http_{name}")
    return pool if isinstance(pool, HTTPConnectionPool) else None