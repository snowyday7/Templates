"""数据库连接池管理模板

提供高性能的数据库连接池管理功能，包括：
- 连接池配置和管理
- 连接健康检查
- 连接池监控
- 多数据库连接池支持
"""

import time
import logging
from typing import Dict, Optional, Any, Callable
from contextlib import contextmanager
from threading import Lock
from dataclasses import dataclass

from sqlalchemy import create_engine, event, pool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import DisconnectionError, SQLAlchemyError


@dataclass
class PoolConfig:
    """连接池配置"""

    # 基础配置
    pool_size: int = 10  # 连接池大小
    max_overflow: int = 20  # 最大溢出连接数
    pool_timeout: int = 30  # 获取连接超时时间（秒）
    pool_recycle: int = 3600  # 连接回收时间（秒）
    pool_pre_ping: bool = True  # 连接前ping检查

    # 高级配置
    pool_reset_on_return: str = "commit"  # 连接返回时的重置方式
    pool_class: str = "QueuePool"  # 连接池类型

    # 监控配置
    enable_monitoring: bool = True  # 启用监控
    log_slow_queries: bool = True  # 记录慢查询
    slow_query_threshold: float = 1.0  # 慢查询阈值（秒）


class PoolMonitor:
    """连接池监控器"""

    def __init__(self):
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "connection_errors": 0,
            "slow_queries": 0,
            "total_queries": 0,
            "avg_query_time": 0.0,
        }
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)

    def record_connection_checkout(self):
        """记录连接检出"""
        with self.lock:
            self.stats["active_connections"] += 1
            self.stats["pool_hits"] += 1

    def record_connection_checkin(self):
        """记录连接检入"""
        with self.lock:
            self.stats["active_connections"] -= 1

    def record_connection_error(self):
        """记录连接错误"""
        with self.lock:
            self.stats["connection_errors"] += 1

    def record_query(self, duration: float):
        """记录查询"""
        with self.lock:
            self.stats["total_queries"] += 1
            # 计算平均查询时间
            total_time = self.stats["avg_query_time"] * (
                self.stats["total_queries"] - 1
            )
            self.stats["avg_query_time"] = (total_time + duration) / self.stats[
                "total_queries"
            ]

    def record_slow_query(self, duration: float, query: str):
        """记录慢查询"""
        with self.lock:
            self.stats["slow_queries"] += 1

        self.logger.warning(
            f"Slow query detected (duration: {duration:.2f}s): {query[:200]}..."
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        with self.lock:
            for key in self.stats:
                if isinstance(self.stats[key], (int, float)):
                    self.stats[key] = 0


class ConnectionPoolManager:
    """连接池管理器"""

    def __init__(self):
        self.pools: Dict[str, Engine] = {}
        self.sessions: Dict[str, sessionmaker] = {}
        self.monitors: Dict[str, PoolMonitor] = {}
        self.configs: Dict[str, PoolConfig] = {}
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)

    def create_pool(
        self, name: str, database_url: str, config: Optional[PoolConfig] = None
    ) -> Engine:
        """创建连接池"""

        config = config or PoolConfig()

        with self.lock:
            if name in self.pools:
                raise ValueError(f"Pool '{name}' already exists")

            # 选择连接池类
            pool_class_map = {
                "QueuePool": QueuePool,
                "NullPool": NullPool,
                "StaticPool": StaticPool,
            }
            pool_class = pool_class_map.get(config.pool_class, QueuePool)

            # 创建引擎
            engine = create_engine(
                database_url,
                poolclass=pool_class,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_timeout=config.pool_timeout,
                pool_recycle=config.pool_recycle,
                pool_pre_ping=config.pool_pre_ping,
                pool_reset_on_return=config.pool_reset_on_return,
                echo=False,  # 通过监控器处理日志
            )

            # 创建会话工厂
            session_factory = sessionmaker(
                bind=engine, autocommit=False, autoflush=False
            )

            # 创建监控器
            monitor = PoolMonitor()

            # 注册事件监听器
            if config.enable_monitoring:
                self._register_events(engine, monitor, config)

            # 保存配置
            self.pools[name] = engine
            self.sessions[name] = session_factory
            self.monitors[name] = monitor
            self.configs[name] = config

            self.logger.info(
                f"Created connection pool '{name}' with {config.pool_size} connections"
            )

            return engine

    def _register_events(
        self, engine: Engine, monitor: PoolMonitor, config: PoolConfig
    ):
        """注册事件监听器"""

        @event.listens_for(engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """连接建立时"""
            monitor.stats["total_connections"] += 1

        @event.listens_for(engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """连接检出时"""
            monitor.record_connection_checkout()

        @event.listens_for(engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """连接检入时"""
            monitor.record_connection_checkin()

        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """查询执行前"""
            context._query_start_time = time.time()

        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """查询执行后"""
            if hasattr(context, "_query_start_time"):
                duration = time.time() - context._query_start_time
                monitor.record_query(duration)

                # 记录慢查询
                if config.log_slow_queries and duration > config.slow_query_threshold:
                    monitor.record_slow_query(duration, statement)

    def get_pool(self, name: str) -> Optional[Engine]:
        """获取连接池"""
        return self.pools.get(name)

    def get_session_factory(self, name: str) -> Optional[sessionmaker]:
        """获取会话工厂"""
        return self.sessions.get(name)

    @contextmanager
    def get_session(self, pool_name: str = "default"):
        """获取数据库会话"""
        session_factory = self.get_session_factory(pool_name)
        if not session_factory:
            raise ValueError(f"Pool '{pool_name}' not found")

        session = session_factory()
        monitor = self.monitors.get(pool_name)

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            if monitor:
                monitor.record_connection_error()
            raise e
        finally:
            session.close()

    def health_check(self, pool_name: str) -> Dict[str, Any]:
        """健康检查"""
        engine = self.get_pool(pool_name)
        if not engine:
            return {"status": "error", "message": f"Pool '{pool_name}' not found"}

        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")

            # 获取连接池状态
            pool = engine.pool
            pool_status = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid(),
            }

            # 获取监控统计
            monitor = self.monitors.get(pool_name)
            stats = monitor.get_stats() if monitor else {}

            return {"status": "healthy", "pool_status": pool_status, "stats": stats}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close_pool(self, name: str):
        """关闭连接池"""
        with self.lock:
            if name in self.pools:
                self.pools[name].dispose()
                del self.pools[name]
                del self.sessions[name]
                del self.monitors[name]
                del self.configs[name]
                self.logger.info(f"Closed connection pool '{name}'")

    def close_all_pools(self):
        """关闭所有连接池"""
        pool_names = list(self.pools.keys())
        for name in pool_names:
            self.close_pool(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有连接池统计信息"""
        return {name: monitor.get_stats() for name, monitor in self.monitors.items()}


class DatabaseConnectionManager:
    """数据库连接管理器（高级封装）"""

    def __init__(self):
        self.pool_manager = ConnectionPoolManager()
        self.default_pool = "default"

    def setup_database(
        self,
        database_url: str,
        pool_name: str = "default",
        config: Optional[PoolConfig] = None,
    ):
        """设置数据库连接"""
        self.pool_manager.create_pool(pool_name, database_url, config)
        if pool_name == "default":
            self.default_pool = pool_name

    def setup_read_replica(
        self,
        database_url: str,
        pool_name: str = "read_replica",
        config: Optional[PoolConfig] = None,
    ):
        """设置读副本数据库"""
        read_config = config or PoolConfig()
        # 读副本通常可以有更多连接
        read_config.pool_size = read_config.pool_size * 2
        self.pool_manager.create_pool(pool_name, database_url, read_config)

    @contextmanager
    def get_write_session(self):
        """获取写会话"""
        with self.pool_manager.get_session(self.default_pool) as session:
            yield session

    @contextmanager
    def get_read_session(self):
        """获取读会话（优先使用读副本）"""
        pool_name = (
            "read_replica"
            if "read_replica" in self.pool_manager.pools
            else self.default_pool
        )
        with self.pool_manager.get_session(pool_name) as session:
            yield session

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """检查所有连接池健康状态"""
        return {
            name: self.pool_manager.health_check(name)
            for name in self.pool_manager.pools.keys()
        }


# 全局连接管理器实例
connection_manager = DatabaseConnectionManager()


# 使用示例
if __name__ == "__main__":
    import os

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 数据库URL
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/mydb"
    )

    # 创建连接池配置
    config = PoolConfig(
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        enable_monitoring=True,
        log_slow_queries=True,
        slow_query_threshold=0.5,
    )

    # 设置数据库连接
    connection_manager.setup_database(database_url, config=config)

    # 使用连接
    try:
        with connection_manager.get_write_session() as session:
            result = session.execute("SELECT 1 as test")
            print(f"Query result: {result.fetchone()}")

        # 健康检查
        health = connection_manager.health_check_all()
        print(f"Health check: {health}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 清理资源
        connection_manager.pool_manager.close_all_pools()
