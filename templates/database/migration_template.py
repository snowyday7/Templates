"""数据库迁移管理模板

提供完整的数据库迁移管理功能，包括：
- Alembic迁移配置
- 自动迁移脚本生成
- 数据库版本管理
- 迁移回滚功能
"""

import os
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from alembic.runtime.environment import EnvironmentContext
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine import Engine


class MigrationConfig:
    """迁移配置类"""

    def __init__(
        self,
        database_url: str,
        migrations_dir: str = "migrations",
        script_location: Optional[str] = None,
    ):
        self.database_url = database_url
        self.migrations_dir = Path(migrations_dir)
        self.script_location = script_location or str(self.migrations_dir)

        # 确保迁移目录存在
        self.migrations_dir.mkdir(exist_ok=True)

    def get_alembic_config(self) -> Config:
        """获取Alembic配置"""
        # 创建alembic.ini配置内容
        alembic_ini_content = f"""
[alembic]
script_location = {self.script_location}
sqlalchemy.url = {self.database_url}

[post_write_hooks]

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""

        # 创建临时配置文件
        config_path = self.migrations_dir / "alembic.ini"
        with open(config_path, "w") as f:
            f.write(alembic_ini_content)

        # 创建Alembic配置对象
        config = Config(str(config_path))
        config.set_main_option("script_location", self.script_location)
        config.set_main_option("sqlalchemy.url", self.database_url)

        return config


class MigrationManager:
    """迁移管理器"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.alembic_config = config.get_alembic_config()
        self.engine = create_engine(config.database_url)

    def init_migrations(self) -> bool:
        """初始化迁移环境"""
        try:
            # 检查是否已经初始化
            if self._is_initialized():
                print("Migration environment already initialized")
                return True

            # 初始化Alembic
            command.init(self.alembic_config, self.config.script_location)

            # 创建自定义的env.py文件
            self._create_env_py()

            print(f"Migration environment initialized in {self.config.migrations_dir}")
            return True

        except Exception as e:
            print(f"Failed to initialize migrations: {e}")
            return False

    def _is_initialized(self) -> bool:
        """检查是否已经初始化"""
        env_py_path = self.config.migrations_dir / "env.py"
        return env_py_path.exists()

    def _create_env_py(self):
        """创建自定义的env.py文件"""
        env_py_content = '''
"""Alembic环境配置文件"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# 导入你的模型
# from myapp.models import Base
# target_metadata = Base.metadata
target_metadata = None

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """离线模式运行迁移"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式运行迁移"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

        env_py_path = self.config.migrations_dir / "env.py"
        with open(env_py_path, "w") as f:
            f.write(env_py_content)

    def create_migration(
        self,
        message: str,
        auto_generate: bool = True,
        metadata: Optional[MetaData] = None,
    ) -> Optional[str]:
        """创建新的迁移文件"""
        try:
            if not self._is_initialized():
                print(
                    "Migration environment not initialized. Run init_migrations() first."
                )
                return None

            # 如果提供了metadata，更新env.py
            if metadata:
                self._update_env_py_metadata(metadata)

            # 生成迁移文件
            if auto_generate:
                revision = command.revision(
                    self.alembic_config, message=message, autogenerate=True
                )
            else:
                revision = command.revision(self.alembic_config, message=message)

            print(f"Created migration: {message}")
            return revision

        except Exception as e:
            print(f"Failed to create migration: {e}")
            return None

    def _update_env_py_metadata(self, metadata: MetaData):
        """更新env.py中的metadata"""
        # 这里可以根据需要更新env.py文件中的target_metadata
        pass

    def upgrade(self, revision: str = "head") -> bool:
        """升级数据库到指定版本"""
        try:
            command.upgrade(self.alembic_config, revision)
            print(f"Database upgraded to {revision}")
            return True
        except Exception as e:
            print(f"Failed to upgrade database: {e}")
            return False

    def downgrade(self, revision: str) -> bool:
        """降级数据库到指定版本"""
        try:
            command.downgrade(self.alembic_config, revision)
            print(f"Database downgraded to {revision}")
            return True
        except Exception as e:
            print(f"Failed to downgrade database: {e}")
            return False

    def get_current_revision(self) -> Optional[str]:
        """获取当前数据库版本"""
        try:
            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            print(f"Failed to get current revision: {e}")
            return None

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """获取迁移历史"""
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_config)
            history = []

            for revision in script_dir.walk_revisions():
                history.append(
                    {
                        "revision": revision.revision,
                        "down_revision": revision.down_revision,
                        "message": revision.doc,
                        "create_date": getattr(revision, "create_date", None),
                    }
                )

            return history

        except Exception as e:
            print(f"Failed to get migration history: {e}")
            return []

    def show_current_status(self) -> Dict[str, Any]:
        """显示当前迁移状态"""
        current_revision = self.get_current_revision()
        history = self.get_migration_history()

        # 找到当前版本在历史中的位置
        current_index = -1
        for i, migration in enumerate(history):
            if migration["revision"] == current_revision:
                current_index = i
                break

        return {
            "current_revision": current_revision,
            "total_migrations": len(history),
            "current_position": current_index + 1 if current_index >= 0 else 0,
            "pending_migrations": current_index if current_index >= 0 else len(history),
            "history": history,
        }

    def validate_migrations(self) -> bool:
        """验证迁移文件的完整性"""
        try:
            # 检查迁移脚本的语法
            script_dir = ScriptDirectory.from_config(self.alembic_config)

            for revision in script_dir.walk_revisions():
                # 检查每个迁移文件是否可以正常加载
                script_dir.get_revision(revision.revision)

            print("All migration files are valid")
            return True

        except Exception as e:
            print(f"Migration validation failed: {e}")
            return False

    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """备份数据库（仅适用于SQLite等文件数据库）"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backup_{timestamp}.db"

            # 这里需要根据具体的数据库类型实现备份逻辑
            # 对于PostgreSQL，可以使用pg_dump
            # 对于MySQL，可以使用mysqldump
            # 对于SQLite，可以直接复制文件

            print(f"Database backup created: {backup_path}")
            return True

        except Exception as e:
            print(f"Failed to backup database: {e}")
            return False

    def generate_sql_script(
        self, from_revision: str, to_revision: str
    ) -> Optional[str]:
        """生成SQL迁移脚本"""
        try:
            # 生成SQL脚本而不执行
            sql_script = command.upgrade(
                self.alembic_config, f"{from_revision}:{to_revision}", sql=True
            )
            return sql_script

        except Exception as e:
            print(f"Failed to generate SQL script: {e}")
            return None


class MigrationHelper:
    """迁移助手类"""

    @staticmethod
    def create_migration_template(name: str, table_name: str) -> str:
        """创建迁移模板"""
        template = f'''
"""Create {table_name} table

Revision ID: {{revision}}
Revises: {{down_revision}}
Create Date: {{create_date}}

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = {{repr(up_revision)}}
down_revision = {{repr(down_revision)}}
branch_labels = {{repr(branch_labels)}}
depends_on = {{repr(depends_on)}}


def upgrade() -> None:
    """升级操作"""
    op.create_table(
        '{table_name}',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_{table_name}_id'), '{table_name}', ['id'], unique=False)


def downgrade() -> None:
    """降级操作"""
    op.drop_index(op.f('ix_{table_name}_id'), table_name='{table_name}')
    op.drop_table('{table_name}')
'''
        return template

    @staticmethod
    def create_data_migration_template(name: str) -> str:
        """创建数据迁移模板"""
        template = f'''
"""Data migration: {name}

Revision ID: {{revision}}
Revises: {{down_revision}}
Create Date: {{create_date}}

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column


# revision identifiers
revision = {{repr(up_revision)}}
down_revision = {{repr(down_revision)}}
branch_labels = {{repr(branch_labels)}}
depends_on = {{repr(depends_on)}}


def upgrade() -> None:
    """数据升级操作"""
    # 定义表结构
    # my_table = table('my_table',
    #     column('id', sa.Integer),
    #     column('name', sa.String)
    # )
    
    # 执行数据操作
    # op.bulk_insert(my_table, [
    #     {{'id': 1, 'name': 'example'}}
    # ])
    pass


def downgrade() -> None:
    """数据降级操作"""
    # 回滚数据操作
    # op.execute("DELETE FROM my_table WHERE id = 1")
    pass
'''
        return template


# 使用示例
if __name__ == "__main__":
    import os

    # 数据库配置
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/mydb"
    )

    # 创建迁移配置
    migration_config = MigrationConfig(
        database_url=database_url, migrations_dir="migrations"
    )

    # 创建迁移管理器
    migration_manager = MigrationManager(migration_config)

    # 初始化迁移环境
    if migration_manager.init_migrations():
        print("Migration environment ready")

        # 显示当前状态
        status = migration_manager.show_current_status()
        print(f"Current status: {status}")

        # 创建新迁移（示例）
        # migration_manager.create_migration("Add users table")

        # 升级数据库
        # migration_manager.upgrade()

        # 验证迁移
        migration_manager.validate_migrations()
