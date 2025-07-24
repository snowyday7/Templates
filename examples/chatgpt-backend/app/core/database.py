# -*- coding: utf-8 -*-
"""
数据库连接和管理

提供数据库连接、会话管理和初始化功能。
"""

import sys
from pathlib import Path
from typing import Generator, Optional
from contextlib import contextmanager

# 添加模板库路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import Engine
from functools import lru_cache

# 导入模板库组件
from src.core.database import BaseDatabase
from src.core.logging import get_logger

# 导入应用组件
from app.core.config import get_settings
from app.models.base import Base
from app.models import User, Conversation, Message, UserQuota, Usage

logger = get_logger(__name__)
settings = get_settings()


class Database(BaseDatabase):
    """
    数据库管理类
    """
    
    def __init__(self):
        super().__init__()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
    
    def initialize(self) -> None:
        """
        初始化数据库连接
        """
        try:
            # 创建数据库引擎
            if settings.DATABASE_URL.startswith("sqlite"):
                # SQLite配置
                self._engine = create_engine(
                    settings.DATABASE_URL,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 20
                    },
                    echo=settings.DATABASE_ECHO,
                    future=True
                )
                
                # 启用SQLite外键约束
                @event.listens_for(self._engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA cache_size=1000")
                    cursor.execute("PRAGMA temp_store=MEMORY")
                    cursor.close()
            
            else:
                # PostgreSQL/MySQL配置
                self._engine = create_engine(
                    settings.DATABASE_URL,
                    pool_size=settings.DATABASE_POOL_SIZE,
                    max_overflow=settings.DATABASE_MAX_OVERFLOW,
                    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
                    pool_recycle=settings.DATABASE_POOL_RECYCLE,
                    pool_pre_ping=True,
                    echo=settings.DATABASE_ECHO,
                    future=True
                )
            
            # 创建会话工厂
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            logger.info(f"Database initialized: {settings.DATABASE_URL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self) -> None:
        """
        创建数据库表
        """
        try:
            if not self._engine:
                raise RuntimeError("Database not initialized")
            
            # 创建所有表
            Base.metadata.create_all(bind=self._engine)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """
        删除数据库表
        """
        try:
            if not self._engine:
                raise RuntimeError("Database not initialized")
            
            Base.metadata.drop_all(bind=self._engine)
            logger.info("Database tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Generator[Session, None, None]:
        """
        获取数据库会话
        """
        if not self._session_factory:
            raise RuntimeError("Database not initialized")
        
        session = self._session_factory()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        """
        获取数据库会话上下文管理器
        """
        if not self._session_factory:
            raise RuntimeError("Database not initialized")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """
        数据库健康检查
        """
        try:
            if not self._engine:
                return False
            
            with self._engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """
        获取数据库连接信息
        """
        if not self._engine:
            return {"status": "not_initialized"}
        
        pool = self._engine.pool
        return {
            "status": "connected",
            "url": str(self._engine.url).replace(self._engine.url.password or "", "***"),
            "pool_size": getattr(pool, 'size', None),
            "checked_in": getattr(pool, 'checkedin', None),
            "checked_out": getattr(pool, 'checkedout', None),
            "overflow": getattr(pool, 'overflow', None)
        }
    
    def close(self) -> None:
        """
        关闭数据库连接
        """
        try:
            if self._engine:
                self._engine.dispose()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")


# 全局数据库实例
_database: Optional[Database] = None


@lru_cache()
def get_database() -> Database:
    """
    获取数据库实例（单例模式）
    """
    global _database
    if _database is None:
        _database = Database()
        _database.initialize()
    return _database


def get_db_session() -> Generator[Session, None, None]:
    """
    获取数据库会话（依赖注入）
    """
    database = get_database()
    yield from database.get_session()


def init_database() -> None:
    """
    初始化数据库
    """
    try:
        database = get_database()
        database.create_tables()
        
        # 创建初始数据
        _create_initial_data()
        
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def _create_initial_data() -> None:
    """
    创建初始数据
    """
    database = get_database()
    
    with database.get_session_context() as session:
        # 检查是否已有管理员用户
        admin_exists = session.query(User).filter(
            User.username.in_(settings.ADMIN_USERS)
        ).first()
        
        if not admin_exists and settings.ADMIN_USERS:
            # 创建管理员用户
            from app.core.security import get_password_hash
            
            for admin_username in settings.ADMIN_USERS:
                admin_user = User(
                    username=admin_username,
                    email=f"{admin_username}@admin.local",
                    password_hash=get_password_hash("admin123"),  # 默认密码
                    is_active=True,
                    is_admin=True,
                    is_vip=True
                )
                session.add(admin_user)
                
                # 创建管理员配额
                admin_quota = UserQuota(
                    user_id=admin_user.id,
                    daily_requests=999999,
                    monthly_requests=999999,
                    daily_tokens=999999999,
                    monthly_tokens=999999999
                )
                session.add(admin_quota)
            
            logger.info(f"Created {len(settings.ADMIN_USERS)} admin users")


def reset_database() -> None:
    """
    重置数据库（删除所有表并重新创建）
    """
    try:
        database = get_database()
        database.drop_tables()
        database.create_tables()
        _create_initial_data()
        
        logger.info("Database reset completed")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise


def backup_database(backup_path: str) -> None:
    """
    备份数据库
    """
    try:
        if settings.DATABASE_URL.startswith("sqlite"):
            # SQLite备份
            import shutil
            from urllib.parse import urlparse
            
            db_path = urlparse(settings.DATABASE_URL).path
            shutil.copy2(db_path, backup_path)
            
        else:
            # PostgreSQL/MySQL备份需要使用相应的工具
            logger.warning("Database backup for non-SQLite databases not implemented")
            
        logger.info(f"Database backed up to: {backup_path}")
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise


def restore_database(backup_path: str) -> None:
    """
    恢复数据库
    """
    try:
        if settings.DATABASE_URL.startswith("sqlite"):
            # SQLite恢复
            import shutil
            from urllib.parse import urlparse
            
            db_path = urlparse(settings.DATABASE_URL).path
            shutil.copy2(backup_path, db_path)
            
            # 重新初始化数据库连接
            global _database
            if _database:
                _database.close()
                _database = None
            
            get_database()  # 重新初始化
            
        else:
            logger.warning("Database restore for non-SQLite databases not implemented")
            
        logger.info(f"Database restored from: {backup_path}")
        
    except Exception as e:
        logger.error(f"Database restore failed: {e}")
        raise