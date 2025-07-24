# 数据库模块使用指南

数据库模块提供了完整的数据库操作功能，包括SQLAlchemy ORM模板、连接池管理和数据迁移等功能。

## 📋 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [基础使用](#基础使用)
- [高级功能](#高级功能)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

## 🚀 快速开始

### 安装依赖

```bash
pip install sqlalchemy alembic psycopg2-binary
```

### 基础配置

```python
from templates.database import DatabaseConfig, DatabaseManager

# 创建数据库配置
config = DatabaseConfig(
    DB_HOST="localhost",
    DB_PORT=5432,
    DB_USER="your_user",
    DB_PASSWORD="your_password",
    DB_NAME="your_database",
    DB_DRIVER="postgresql+psycopg2"
)

# 创建数据库管理器
db_manager = DatabaseManager(config)
```

## ⚙️ 配置说明

### DatabaseConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `DB_HOST` | str | "localhost" | 数据库主机地址 |
| `DB_PORT` | int | 5432 | 数据库端口 |
| `DB_USER` | str | "postgres" | 数据库用户名 |
| `DB_PASSWORD` | str | "password" | 数据库密码 |
| `DB_NAME` | str | "myapp" | 数据库名称 |
| `DB_DRIVER` | str | "postgresql+psycopg2" | 数据库驱动 |
| `DB_POOL_SIZE` | int | 10 | 连接池大小 |
| `DB_MAX_OVERFLOW` | int | 20 | 最大溢出连接数 |
| `DB_POOL_TIMEOUT` | int | 30 | 连接超时时间(秒) |
| `DB_POOL_RECYCLE` | int | 3600 | 连接回收时间(秒) |
| `DB_ECHO` | bool | False | 是否打印SQL语句 |

### 环境变量配置

创建 `.env` 文件：

```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=myuser
DB_PASSWORD=mypassword
DB_NAME=mydatabase
DB_DRIVER=postgresql+psycopg2
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_ECHO=false
```

## 💻 基础使用

### 1. 创建模型

```python
from templates.database import BaseModel
from sqlalchemy import Column, String, Integer, Boolean

class User(BaseModel):
    __tablename__ = "users"
    
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100))
    is_superuser = Column(Boolean, default=False)
```

### 2. 数据库操作

```python
# 创建表
db_manager.create_tables()

# 使用会话
with db_manager.get_session() as session:
    # 创建用户
    user = User.create(
        session,
        username="john_doe",
        email="john@example.com",
        full_name="John Doe"
    )
    
    # 查询用户
    user = User.get_by_id(session, 1)
    users = User.get_all(session, skip=0, limit=10)
    
    # 更新用户
    user.update(full_name="John Smith")
    
    # 删除用户（软删除）
    user.delete(session, soft_delete=True)
```

### 3. CRUD操作

```python
from templates.database import CRUDBase

class UserCRUD(CRUDBase):
    def __init__(self):
        super().__init__(User)
    
    def get_by_username(self, session, username: str):
        return session.query(self.model).filter(
            self.model.username == username,
            self.model.is_active == True
        ).first()

# 使用CRUD
user_crud = UserCRUD()

with db_manager.get_session() as session:
    # 创建
    user_data = {
        "username": "jane_doe",
        "email": "jane@example.com",
        "full_name": "Jane Doe"
    }
    user = user_crud.create(session, user_data)
    
    # 查询
    user = user_crud.get_by_username(session, "jane_doe")
    
    # 更新
    updated_user = user_crud.update(session, user, {"full_name": "Jane Smith"})
    
    # 删除
    user_crud.delete(session, user.id)
```

## 🔧 高级功能

### 1. 连接池管理

```python
from templates.database import ConnectionPoolManager

# 创建连接池管理器
pool_manager = ConnectionPoolManager(
    database_url="postgresql://user:pass@localhost/db",
    pool_size=20,
    max_overflow=30,
    pool_timeout=60
)

# 获取连接
with pool_manager.get_connection() as conn:
    result = conn.execute("SELECT * FROM users")
    for row in result:
        print(row)
```

### 2. 数据迁移

```python
from templates.database import MigrationManager

# 创建迁移管理器
migration_manager = MigrationManager(
    database_url="postgresql://user:pass@localhost/db",
    script_location="migrations"
)

# 初始化迁移环境
migration_manager.init_migration()

# 生成迁移脚本
migration_manager.generate_migration("add_user_table")

# 执行迁移
migration_manager.upgrade()

# 回滚迁移
migration_manager.downgrade()
```

### 3. 多数据库支持

```python
# 主数据库配置
master_config = DatabaseConfig(
    DB_HOST="master.db.com",
    DB_NAME="master_db"
)

# 从数据库配置
slave_config = DatabaseConfig(
    DB_HOST="slave.db.com",
    DB_NAME="slave_db"
)

# 创建多个数据库管理器
master_db = DatabaseManager(master_config)
slave_db = DatabaseManager(slave_config)

# 读写分离
def create_user(user_data):
    with master_db.get_session() as session:
        return User.create(session, **user_data)

def get_users():
    with slave_db.get_session() as session:
        return User.get_all(session)
```

## 📝 最佳实践

### 1. 模型设计

```python
# 好的实践
class User(BaseModel):
    __tablename__ = "users"
    
    # 使用合适的字段长度
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    
    # 添加索引提高查询性能
    __table_args__ = (
        Index("idx_user_email_active", "email", "is_active"),
        Index("idx_user_username_active", "username", "is_active"),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"
```

### 2. 会话管理

```python
# 推荐：使用上下文管理器
with db_manager.get_session() as session:
    user = User.create(session, username="test")
    # 自动提交和关闭

# 避免：手动管理会话
session = db_manager.SessionLocal()
try:
    user = User.create(session, username="test")
    session.commit()
except Exception:
    session.rollback()
finally:
    session.close()
```

### 3. 错误处理

```python
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

def create_user_safe(user_data):
    try:
        with db_manager.get_session() as session:
            return User.create(session, **user_data)
    except IntegrityError as e:
        # 处理唯一约束违反
        raise ValueError(f"用户已存在: {e}")
    except SQLAlchemyError as e:
        # 处理其他数据库错误
        raise RuntimeError(f"数据库操作失败: {e}")
```

### 4. 性能优化

```python
# 使用批量操作
def create_users_batch(users_data):
    with db_manager.get_session() as session:
        users = [User(**data) for data in users_data]
        session.add_all(users)
        session.flush()
        return users

# 使用预加载避免N+1查询
from sqlalchemy.orm import joinedload

def get_users_with_posts():
    with db_manager.get_session() as session:
        return session.query(User).options(
            joinedload(User.posts)
        ).all()
```

## ❓ 常见问题

### Q: 如何处理数据库连接超时？

A: 配置合适的连接池参数：

```python
config = DatabaseConfig(
    DB_POOL_TIMEOUT=30,  # 连接超时30秒
    DB_POOL_RECYCLE=3600,  # 1小时回收连接
    DB_POOL_SIZE=10,  # 连接池大小
    DB_MAX_OVERFLOW=20  # 最大溢出连接
)
```

### Q: 如何进行数据库迁移？

 A: 使用内置的迁移管理器：

```bash
# 初始化迁移
python -c "from templates.database import MigrationManager; MigrationManager().init_migration()"

# 生成迁移
python -c "from templates.database import MigrationManager; MigrationManager().generate_migration('description')"

# 执行迁移
python -c "from templates.database import MigrationManager; MigrationManager().upgrade()"
```

### Q: 如何实现读写分离？

A: 创建多个数据库管理器实例：

```python
# 写库
write_db = DatabaseManager(write_config)

# 读库
read_db = DatabaseManager(read_config)

# 写操作使用write_db，读操作使用read_db
```

### Q: 如何处理大量数据的查询？

A: 使用分页和流式查询：

```python
# 分页查询
def get_users_paginated(page=1, size=100):
    with db_manager.get_session() as session:
        return User.get_all(session, skip=(page-1)*size, limit=size)

# 流式查询
def process_all_users():
    with db_manager.get_session() as session:
        for user in session.query(User).yield_per(1000):
            # 处理单个用户
            process_user(user)
```

## 📚 相关文档

- [API开发模块使用指南](api.md)
- [认证授权模块使用指南](auth.md)
- [项目结构最佳实践](best-practices/project-structure.md)
- [性能优化建议](best-practices/performance.md)

---

如有其他问题，请查看 [GitHub Issues](https://github.com/your-username/python-backend-templates/issues) 或提交新的问题。