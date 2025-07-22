"""SQLAlchemy ORM模板

提供完整的SQLAlchemy ORM使用模板，包括：
- 数据库连接配置
- 模型基类定义
- 会话管理
- 常用查询方法
"""

from typing import Optional, List, Dict, Any, Type, TypeVar
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from pydantic_settings import BaseSettings

# 类型变量
ModelType = TypeVar("ModelType", bound="BaseModel")


class DatabaseConfig(BaseSettings):
    """数据库配置类"""

    # 数据库连接配置
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "password"
    DB_NAME: str = "myapp"
    DB_DRIVER: str = "postgresql+psycopg2"

    # 连接池配置
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600

    # 其他配置
    DB_ECHO: bool = False
    DB_ECHO_POOL: bool = False

    class Config:
        env_file = ".env"

    @property
    def database_url(self) -> str:
        """构建数据库连接URL"""
        return (
            f"{self.DB_DRIVER}://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.SessionLocal = None
        self._setup_database()

    def _setup_database(self):
        """设置数据库连接"""
        self.engine = create_engine(
            self.config.database_url,
            poolclass=QueuePool,
            pool_size=self.config.DB_POOL_SIZE,
            max_overflow=self.config.DB_MAX_OVERFLOW,
            pool_timeout=self.config.DB_POOL_TIMEOUT,
            pool_recycle=self.config.DB_POOL_RECYCLE,
            echo=self.config.DB_ECHO,
            echo_pool=self.config.DB_ECHO_POOL,
        )

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """删除所有表"""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        """获取数据库会话上下文管理器"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


# 创建基类
Base = declarative_base()


class BaseModel(Base):
    """模型基类"""

    __abstract__ = True

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    is_active = Column(Boolean, default=True, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }

    def update(self, **kwargs):
        """更新模型属性"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()

    @classmethod
    def create(cls, session: Session, **kwargs) -> "BaseModel":
        """创建新实例"""
        instance = cls(**kwargs)
        session.add(instance)
        session.flush()
        return instance

    @classmethod
    def get_by_id(cls, session: Session, id: int) -> Optional["BaseModel"]:
        """根据ID获取实例"""
        return session.query(cls).filter(cls.id == id, cls.is_active == True).first()

    @classmethod
    def get_all(
        cls, session: Session, skip: int = 0, limit: int = 100
    ) -> List["BaseModel"]:
        """获取所有实例"""
        return (
            session.query(cls)
            .filter(cls.is_active == True)
            .offset(skip)
            .limit(limit)
            .all()
        )

    @classmethod
    def count(cls, session: Session) -> int:
        """获取总数"""
        return session.query(cls).filter(cls.is_active == True).count()

    def delete(self, session: Session, soft_delete: bool = True):
        """删除实例"""
        if soft_delete:
            self.is_active = False
            self.updated_at = datetime.utcnow()
        else:
            session.delete(self)


class CRUDBase:
    """CRUD操作基类"""

    def __init__(self, model: Type[ModelType]):
        self.model = model

    def create(self, session: Session, obj_in: Dict[str, Any]) -> ModelType:
        """创建"""
        db_obj = self.model(**obj_in)
        session.add(db_obj)
        session.flush()
        return db_obj

    def get(self, session: Session, id: int) -> Optional[ModelType]:
        """根据ID获取"""
        return (
            session.query(self.model)
            .filter(self.model.id == id, self.model.is_active == True)
            .first()
        )

    def get_multi(
        self, session: Session, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        """获取多个"""
        return (
            session.query(self.model)
            .filter(self.model.is_active == True)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def update(
        self, session: Session, db_obj: ModelType, obj_in: Dict[str, Any]
    ) -> ModelType:
        """更新"""
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        db_obj.updated_at = datetime.utcnow()
        session.flush()
        return db_obj

    def delete(self, session: Session, id: int, soft_delete: bool = True) -> bool:
        """删除"""
        obj = self.get(session, id)
        if obj:
            if soft_delete:
                obj.is_active = False
                obj.updated_at = datetime.utcnow()
            else:
                session.delete(obj)
            return True
        return False


# 示例模型
class User(BaseModel):
    """用户模型示例"""

    __tablename__ = "users"

    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_superuser = Column(Boolean, default=False)

    # 索引
    __table_args__ = (
        Index("idx_user_email_active", "email", "is_active"),
        Index("idx_user_username_active", "username", "is_active"),
    )

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Post(BaseModel):
    """文章模型示例"""

    __tablename__ = "posts"

    title = Column(String(200), nullable=False, index=True)
    content = Column(Text)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_published = Column(Boolean, default=False)

    # 关系
    author = relationship("User", backref="posts")

    # 索引和约束
    __table_args__ = (
        Index("idx_post_author_published", "author_id", "is_published"),
        Index("idx_post_title_active", "title", "is_active"),
    )

    def __repr__(self):
        return f"<Post(id={self.id}, title='{self.title}')>"


# 全局数据库管理器实例（延迟初始化）
db_manager = None

def get_db_manager() -> DatabaseManager:
    """获取数据库管理器实例（延迟初始化）"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def create_database_session() -> Session:
    """创建数据库会话"""
    return db_manager.SessionLocal()


def get_db_session():
    """获取数据库会话依赖（用于FastAPI等框架）"""
    session = create_database_session()
    try:
        yield session
    finally:
        session.close()


# CRUD实例
user_crud = CRUDBase(User)
post_crud = CRUDBase(Post)


# 使用示例
if __name__ == "__main__":
    # 创建表
    db_manager.create_tables()

    # 使用会话
    with db_manager.get_session() as session:
        # 创建用户
        user_data = {
            "username": "john_doe",
            "email": "john@example.com",
            "hashed_password": "hashed_password_here",
            "full_name": "John Doe",
        }
        user = user_crud.create(session, user_data)
        print(f"Created user: {user}")

        # 创建文章
        post_data = {
            "title": "My First Post",
            "content": "This is the content of my first post.",
            "author_id": user.id,
            "is_published": True,
        }
        post = post_crud.create(session, post_data)
        print(f"Created post: {post}")

        # 查询
        users = user_crud.get_multi(session)
        print(f"All users: {users}")