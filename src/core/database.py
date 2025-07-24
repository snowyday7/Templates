#!/usr/bin/env python3
"""
数据库连接和会话管理模块

提供统一的数据库访问接口，支持：
1. SQLAlchemy ORM
2. 连接池管理
3. 事务管理
4. 读写分离
5. 数据库迁移
6. 健康检查
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any

from sqlalchemy import (
    create_engine,
    event,
    pool,
    text,
    MetaData,
    inspect,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from alembic import command
from alembic.config import Config
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config import get_settings, get_database_url


# 配置日志
logger = logging.getLogger(__name__)

# 创建基础模型类
Base = declarative_base()

# 元数据约定
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
Base.metadata = MetaData(naming_convention=convention)


class DatabaseManager:
    """
    数据库管理器
    
    负责管理数据库连接、会话和事务
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._engine: Optional[AsyncEngine] = None
        self._read_engine: Optional[AsyncEngine] = None
        self._sync_engine = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._read_session_factory: Optional[async_sessionmaker] = None
        self._sync_session_factory = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        初始化数据库连接
        """
        if self._initialized:
            return
        
        try:
            # 创建主数据库引擎（读写）
            database_url = get_database_url(self.settings)
            self._engine = await self._create_async_engine(database_url)
            
            # 创建只读数据库引擎（如果配置了）
            if self.settings.read_database_url:
                self._read_engine = await self._create_async_engine(
                    str(self.settings.read_database_url),
                    read_only=True
                )
            else:
                self._read_engine = self._engine
            
            # 创建同步引擎（用于迁移等操作）
            sync_database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
            self._sync_engine = create_engine(
                sync_database_url,
                poolclass=QueuePool,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow,
                pool_timeout=self.settings.database_pool_timeout,
                pool_recycle=self.settings.database_pool_recycle,
                echo=self.settings.database_echo,
            )
            
            # 创建会话工厂
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
            
            self._read_session_factory = async_sessionmaker(
                bind=self._read_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False,
            )
            
            self._sync_session_factory = sessionmaker(
                bind=self._sync_engine,
                autoflush=True,
                autocommit=False,
            )
            
            # 注册事件监听器
            self._register_event_listeners()
            
            # 测试连接
            await self._test_connection()
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _create_async_engine(self, database_url: str, read_only: bool = False) -> AsyncEngine:
        """
        创建异步数据库引擎
        
        Args:
            database_url: 数据库连接URL
            read_only: 是否为只读连接
        
        Returns:
            AsyncEngine: 异步数据库引擎
        """
        # 转换为异步URL
        if not database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        # 引擎配置
        engine_kwargs = {
            "url": database_url,
            "echo": self.settings.database_echo,
            "pool_size": self.settings.database_pool_size,
            "max_overflow": self.settings.database_max_overflow,
            "pool_timeout": self.settings.database_pool_timeout,
            "pool_recycle": self.settings.database_pool_recycle,
            "poolclass": QueuePool,
        }
        
        # 测试环境使用NullPool
        if self.settings.testing:
            engine_kwargs["poolclass"] = NullPool
        
        # 只读连接的特殊配置
        if read_only:
            engine_kwargs["connect_args"] = {
                "server_settings": {
                    "default_transaction_read_only": "on"
                }
            }
        
        return create_async_engine(**engine_kwargs)
    
    def _register_event_listeners(self) -> None:
        """
        注册数据库事件监听器
        """
        # 连接事件
        @event.listens_for(self._engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """设置数据库连接参数"""
            if "sqlite" in str(self._engine.url):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        
        # 连接池事件
        @event.listens_for(self._engine.sync_engine, "pool_connect")
        def set_pool_connect(dbapi_connection, connection_record):
            """连接池连接事件"""
            logger.debug("New database connection established")
        
        @event.listens_for(self._engine.sync_engine, "pool_checkout")
        def set_pool_checkout(dbapi_connection, connection_record, connection_proxy):
            """连接池检出事件"""
            logger.debug("Database connection checked out from pool")
        
        @event.listens_for(self._engine.sync_engine, "pool_checkin")
        def set_pool_checkin(dbapi_connection, connection_record):
            """连接池检入事件"""
            logger.debug("Database connection checked in to pool")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((SQLAlchemyError, DisconnectionError)),
    )
    async def _test_connection(self) -> None:
        """
        测试数据库连接
        """
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def close(self) -> None:
        """
        关闭数据库连接
        """
        if self._engine:
            await self._engine.dispose()
        if self._read_engine and self._read_engine != self._engine:
            await self._read_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()
        
        self._initialized = False
        logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self, read_only: bool = False) -> AsyncGenerator[AsyncSession, None]:
        """
        获取数据库会话（上下文管理器）
        
        Args:
            read_only: 是否为只读会话
        
        Yields:
            AsyncSession: 数据库会话
        """
        if not self._initialized:
            await self.initialize()
        
        session_factory = self._read_session_factory if read_only else self._session_factory
        session = session_factory()
        
        try:
            yield session
            if not read_only:
                await session.commit()
        except Exception as e:
            if not read_only:
                await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    async def get_session_instance(self, read_only: bool = False) -> AsyncSession:
        """
        获取数据库会话实例
        
        Args:
            read_only: 是否为只读会话
        
        Returns:
            AsyncSession: 数据库会话
        """
        if not self._initialized:
            await self.initialize()
        
        session_factory = self._read_session_factory if read_only else self._session_factory
        return session_factory()
    
    def get_sync_session(self) -> Session:
        """
        获取同步数据库会话
        
        Returns:
            Session: 同步数据库会话
        """
        if not self._sync_session_factory:
            raise RuntimeError("Database not initialized")
        
        return self._sync_session_factory()
    
    async def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行原始SQL
        
        Args:
            sql: SQL语句
            params: 参数
        
        Returns:
            Any: 执行结果
        """
        async with self.get_session() as session:
            result = await session.execute(text(sql), params or {})
            return result
    
    async def health_check(self) -> Dict[str, Any]:
        """
        数据库健康检查
        
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            # 检查主数据库
            start_time = asyncio.get_event_loop().time()
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            main_db_time = asyncio.get_event_loop().time() - start_time
            
            # 检查只读数据库
            read_db_time = None
            if self._read_engine != self._engine:
                start_time = asyncio.get_event_loop().time()
                async with self.get_session(read_only=True) as session:
                    await session.execute(text("SELECT 1"))
                read_db_time = asyncio.get_event_loop().time() - start_time
            
            # 获取连接池状态
            pool_status = {
                "size": self._engine.pool.size(),
                "checked_in": self._engine.pool.checkedin(),
                "checked_out": self._engine.pool.checkedout(),
                "overflow": self._engine.pool.overflow(),
            }
            
            return {
                "status": "healthy",
                "main_db_response_time": main_db_time,
                "read_db_response_time": read_db_time,
                "pool_status": pool_status,
                "timestamp": asyncio.get_event_loop().time(),
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time(),
            }
    
    def run_migrations(self, revision: str = "head") -> None:
        """
        运行数据库迁移
        
        Args:
            revision: 迁移版本
        """
        try:
            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, revision)
            logger.info(f"Database migration to {revision} completed")
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            raise
    
    def create_revision(self, message: str, autogenerate: bool = True) -> None:
        """
        创建数据库迁移版本
        
        Args:
            message: 迁移消息
            autogenerate: 是否自动生成
        """
        try:
            alembic_cfg = Config("alembic.ini")
            command.revision(alembic_cfg, message=message, autogenerate=autogenerate)
            logger.info(f"Database revision '{message}' created")
        except Exception as e:
            logger.error(f"Failed to create database revision: {e}")
            raise
    
    async def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库信息
        
        Returns:
            Dict[str, Any]: 数据库信息
        """
        try:
            async with self.get_session() as session:
                # 获取数据库版本
                version_result = await session.execute(text("SELECT version()"))
                version = version_result.scalar()
                
                # 获取数据库大小
                size_result = await session.execute(
                    text("SELECT pg_size_pretty(pg_database_size(current_database()))")
                )
                size = size_result.scalar()
                
                # 获取连接数
                connections_result = await session.execute(
                    text("SELECT count(*) FROM pg_stat_activity")
                )
                connections = connections_result.scalar()
                
                return {
                    "version": version,
                    "size": size,
                    "connections": connections,
                    "url": str(self._engine.url).replace(self._engine.url.password or "", "***"),
                }
                
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}


# 全局数据库管理器实例
db_manager = DatabaseManager()


# 依赖注入函数
async def get_db_session(read_only: bool = False) -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话（依赖注入）
    
    Args:
        read_only: 是否为只读会话
    
    Yields:
        AsyncSession: 数据库会话
    """
    async with db_manager.get_session(read_only=read_only) as session:
        yield session


async def get_read_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    获取只读数据库会话（依赖注入）
    
    Yields:
        AsyncSession: 只读数据库会话
    """
    async with db_manager.get_session(read_only=True) as session:
        yield session


# 事务装饰器
def transactional(read_only: bool = False):
    """
    事务装饰器
    
    Args:
        read_only: 是否为只读事务
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with db_manager.get_session(read_only=read_only) as session:
                # 将session注入到函数参数中
                if 'session' not in kwargs:
                    kwargs['session'] = session
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# 数据库初始化函数
async def init_database() -> None:
    """
    初始化数据库
    """
    await db_manager.initialize()


# 数据库关闭函数
async def close_database() -> None:
    """
    关闭数据库连接
    """
    await db_manager.close()


# 数据库健康检查函数
async def database_health_check() -> Dict[str, Any]:
    """
    数据库健康检查
    
    Returns:
        Dict[str, Any]: 健康检查结果
    """
    return await db_manager.health_check()