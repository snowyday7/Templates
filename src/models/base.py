#!/usr/bin/env python3
"""
基础模型模块

提供所有数据模型的基础类和混入类，包括：
1. 基础模型类
2. 时间戳混入
3. 软删除混入
4. 审计混入
5. 版本控制混入
6. 通用查询方法
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from datetime import datetime

from sqlalchemy import Column, Integer, DateTime, Boolean, String, Text, event
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.inspection import inspect

from ..core.config import get_settings


# 创建基础模型类
Base = declarative_base()

# 类型变量
ModelType = TypeVar("ModelType", bound="BaseModel")


class BaseModel(Base):
    """
    基础模型类
    
    所有数据模型的基类，提供通用的字段和方法
    """
    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True, comment="主键ID")
    
    @declared_attr
    def __tablename__(cls) -> str:
        """自动生成表名"""
        # 将类名转换为下划线命名
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        转换为字典
        
        Args:
            exclude: 要排除的字段列表
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        exclude = exclude or []
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                # 处理日期时间类型
                if isinstance(value, datetime):
                    value = value.isoformat()
                result[column.name] = value
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude: Optional[List[str]] = None) -> None:
        """
        从字典更新属性
        
        Args:
            data: 数据字典
            exclude: 要排除的字段列表
        """
        exclude = exclude or ['id', 'created_at']
        
        for key, value in data.items():
            if key not in exclude and hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    async def get_by_id(
        cls: Type[ModelType],
        db: AsyncSession,
        id: int
    ) -> Optional[ModelType]:
        """
        根据ID获取记录
        
        Args:
            db: 数据库会话
            id: 记录ID
        
        Returns:
            Optional[ModelType]: 记录对象或None
        """
        from sqlalchemy import select
        result = await db.execute(select(cls).where(cls.id == id))
        return result.scalar_one_or_none()
    
    @classmethod
    async def get_all(
        cls: Type[ModelType],
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """
        获取所有记录
        
        Args:
            db: 数据库会话
            skip: 跳过的记录数
            limit: 限制的记录数
        
        Returns:
            List[ModelType]: 记录列表
        """
        from sqlalchemy import select
        result = await db.execute(
            select(cls).offset(skip).limit(limit)
        )
        return result.scalars().all()
    
    @classmethod
    async def count(
        cls: Type[ModelType],
        db: AsyncSession
    ) -> int:
        """
        获取记录总数
        
        Args:
            db: 数据库会话
        
        Returns:
            int: 记录总数
        """
        from sqlalchemy import select, func
        result = await db.execute(select(func.count(cls.id)))
        return result.scalar()
    
    async def save(self, db: AsyncSession) -> None:
        """
        保存记录
        
        Args:
            db: 数据库会话
        """
        db.add(self)
        await db.commit()
        await db.refresh(self)
    
    async def delete(self, db: AsyncSession) -> None:
        """
        删除记录
        
        Args:
            db: 数据库会话
        """
        await db.delete(self)
        await db.commit()
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"<{self.__class__.__name__}(id={self.id})>"


class TimestampMixin:
    """
    时间戳混入类
    
    提供创建时间和更新时间字段
    """
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="更新时间"
    )


class SoftDeleteMixin:
    """
    软删除混入类
    
    提供软删除功能
    """
    
    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="是否已删除"
    )
    
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="删除时间"
    )
    
    async def soft_delete(self, db: AsyncSession) -> None:
        """
        软删除记录
        
        Args:
            db: 数据库会话
        """
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        await self.save(db)
    
    async def restore(self, db: AsyncSession) -> None:
        """
        恢复删除的记录
        
        Args:
            db: 数据库会话
        """
        self.is_deleted = False
        self.deleted_at = None
        await self.save(db)
    
    @classmethod
    async def get_active(
        cls: Type[ModelType],
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """
        获取未删除的记录
        
        Args:
            db: 数据库会话
            skip: 跳过的记录数
            limit: 限制的记录数
        
        Returns:
            List[ModelType]: 记录列表
        """
        from sqlalchemy import select
        result = await db.execute(
            select(cls)
            .where(cls.is_deleted == False)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    @classmethod
    async def count_active(
        cls: Type[ModelType],
        db: AsyncSession
    ) -> int:
        """
        获取未删除记录的总数
        
        Args:
            db: 数据库会话
        
        Returns:
            int: 记录总数
        """
        from sqlalchemy import select, func
        result = await db.execute(
            select(func.count(cls.id))
            .where(cls.is_deleted == False)
        )
        return result.scalar()


class AuditMixin:
    """
    审计混入类
    
    提供创建者和更新者字段
    """
    
    created_by = Column(
        Integer,
        nullable=True,
        comment="创建者ID"
    )
    
    updated_by = Column(
        Integer,
        nullable=True,
        comment="更新者ID"
    )


class VersionMixin:
    """
    版本控制混入类
    
    提供乐观锁版本控制
    """
    
    version = Column(
        Integer,
        default=1,
        nullable=False,
        comment="版本号"
    )
    
    def increment_version(self) -> None:
        """增加版本号"""
        self.version += 1


class MetadataMixin:
    """
    元数据混入类
    
    提供额外的元数据字段
    """
    
    metadata_json = Column(
        Text,
        nullable=True,
        comment="JSON格式的元数据"
    )
    
    tags = Column(
        String(500),
        nullable=True,
        comment="标签（逗号分隔）"
    )
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取元数据
        
        Returns:
            Dict[str, Any]: 元数据字典
        """
        if not self.metadata_json:
            return {}
        
        import json
        try:
            return json.loads(self.metadata_json)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        设置元数据
        
        Args:
            metadata: 元数据字典
        """
        import json
        self.metadata_json = json.dumps(metadata, ensure_ascii=False)
    
    def get_tags(self) -> List[str]:
        """
        获取标签列表
        
        Returns:
            List[str]: 标签列表
        """
        if not self.tags:
            return []
        return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def set_tags(self, tags: List[str]) -> None:
        """
        设置标签
        
        Args:
            tags: 标签列表
        """
        self.tags = ','.join(tags) if tags else None


# =============================================================================
# 事件监听器
# =============================================================================

@event.listens_for(BaseModel, 'before_insert', propagate=True)
def before_insert(mapper, connection, target):
    """
    插入前事件监听器
    
    Args:
        mapper: SQLAlchemy映射器
        connection: 数据库连接
        target: 目标对象
    """
    # 设置创建时间
    if hasattr(target, 'created_at') and target.created_at is None:
        target.created_at = datetime.utcnow()
    
    # 设置更新时间
    if hasattr(target, 'updated_at') and target.updated_at is None:
        target.updated_at = datetime.utcnow()


@event.listens_for(BaseModel, 'before_update', propagate=True)
def before_update(mapper, connection, target):
    """
    更新前事件监听器
    
    Args:
        mapper: SQLAlchemy映射器
        connection: 数据库连接
        target: 目标对象
    """
    # 更新时间
    if hasattr(target, 'updated_at'):
        target.updated_at = datetime.utcnow()
    
    # 增加版本号
    if hasattr(target, 'version'):
        target.increment_version()


# =============================================================================
# 查询构建器
# =============================================================================

class QueryBuilder:
    """
    查询构建器
    
    提供便捷的查询构建方法
    """
    
    def __init__(self, model_class: Type[BaseModel]):
        self.model_class = model_class
        self.query = None
    
    def filter_by(self, **kwargs) -> "QueryBuilder":
        """
        按条件过滤
        
        Args:
            **kwargs: 过滤条件
        
        Returns:
            QueryBuilder: 查询构建器
        """
        from sqlalchemy import select
        
        if self.query is None:
            self.query = select(self.model_class)
        
        for key, value in kwargs.items():
            if hasattr(self.model_class, key):
                column = getattr(self.model_class, key)
                self.query = self.query.where(column == value)
        
        return self
    
    def order_by(self, *columns) -> "QueryBuilder":
        """
        排序
        
        Args:
            *columns: 排序列
        
        Returns:
            QueryBuilder: 查询构建器
        """
        if self.query is not None:
            self.query = self.query.order_by(*columns)
        return self
    
    def limit(self, limit: int) -> "QueryBuilder":
        """
        限制结果数量
        
        Args:
            limit: 限制数量
        
        Returns:
            QueryBuilder: 查询构建器
        """
        if self.query is not None:
            self.query = self.query.limit(limit)
        return self
    
    def offset(self, offset: int) -> "QueryBuilder":
        """
        跳过记录数
        
        Args:
            offset: 跳过数量
        
        Returns:
            QueryBuilder: 查询构建器
        """
        if self.query is not None:
            self.query = self.query.offset(offset)
        return self
    
    async def all(self, db: AsyncSession) -> List[BaseModel]:
        """
        获取所有结果
        
        Args:
            db: 数据库会话
        
        Returns:
            List[BaseModel]: 结果列表
        """
        if self.query is None:
            return []
        
        result = await db.execute(self.query)
        return result.scalars().all()
    
    async def first(self, db: AsyncSession) -> Optional[BaseModel]:
        """
        获取第一个结果
        
        Args:
            db: 数据库会话
        
        Returns:
            Optional[BaseModel]: 结果对象或None
        """
        if self.query is None:
            return None
        
        result = await db.execute(self.query)
        return result.scalar_one_or_none()
    
    async def count(self, db: AsyncSession) -> int:
        """
        获取结果数量
        
        Args:
            db: 数据库会话
        
        Returns:
            int: 结果数量
        """
        if self.query is None:
            return 0
        
        from sqlalchemy import func, select
        count_query = select(func.count()).select_from(self.query.subquery())
        result = await db.execute(count_query)
        return result.scalar()


# =============================================================================
# 便捷函数
# =============================================================================

def create_query_builder(model_class: Type[BaseModel]) -> QueryBuilder:
    """
    创建查询构建器
    
    Args:
        model_class: 模型类
    
    Returns:
        QueryBuilder: 查询构建器
    """
    return QueryBuilder(model_class)


async def get_or_create(
    db: AsyncSession,
    model_class: Type[ModelType],
    defaults: Optional[Dict[str, Any]] = None,
    **kwargs
) -> tuple[ModelType, bool]:
    """
    获取或创建记录
    
    Args:
        db: 数据库会话
        model_class: 模型类
        defaults: 默认值
        **kwargs: 查询条件
    
    Returns:
        tuple[ModelType, bool]: (记录对象, 是否新创建)
    """
    from sqlalchemy import select
    
    # 构建查询条件
    query = select(model_class)
    for key, value in kwargs.items():
        if hasattr(model_class, key):
            column = getattr(model_class, key)
            query = query.where(column == value)
    
    # 查询记录
    result = await db.execute(query)
    instance = result.scalar_one_or_none()
    
    if instance:
        return instance, False
    
    # 创建新记录
    create_data = kwargs.copy()
    if defaults:
        create_data.update(defaults)
    
    instance = model_class(**create_data)
    await instance.save(db)
    
    return instance, True


async def bulk_create(
    db: AsyncSession,
    model_class: Type[ModelType],
    data_list: List[Dict[str, Any]]
) -> List[ModelType]:
    """
    批量创建记录
    
    Args:
        db: 数据库会话
        model_class: 模型类
        data_list: 数据列表
    
    Returns:
        List[ModelType]: 创建的记录列表
    """
    instances = []
    for data in data_list:
        instance = model_class(**data)
        instances.append(instance)
        db.add(instance)
    
    await db.commit()
    
    # 刷新所有实例
    for instance in instances:
        await db.refresh(instance)
    
    return instances