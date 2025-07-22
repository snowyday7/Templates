"""数据库模块模板

提供各种数据库操作的模板代码，包括：
- SQLAlchemy ORM模板
- 数据库连接池管理
- 数据迁移脚本
- 多数据库支持
"""

from .sqlalchemy_template import (
    DatabaseConfig,
    DatabaseManager,
    Base,
    BaseModel,
)
from .connection_pool import ConnectionPoolManager
from .migration_template import MigrationManager

__all__ = [
    "DatabaseConfig",
    "DatabaseManager",
    "Base",
    "BaseModel",
    "ConnectionPoolManager",
    "MigrationManager",
]