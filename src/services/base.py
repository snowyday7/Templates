#!/usr/bin/env python3
"""
服务层基础类

提供所有服务类的基础功能，包括：
1. 数据库会话管理
2. 缓存操作
3. 通用CRUD操作
4. 事务管理
5. 错误处理
"""

from typing import Optional, List, Dict, Any, Type, TypeVar, Generic, Union
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload, joinedload
from pydantic import BaseModel

from ..core.database import get_db_session
from ..core.cache import CacheManager
from ..core.exceptions import (
    ValidationException,
    ResourceNotFoundException,
    ConflictException,
    InternalServerException
)
from ..models.base import BaseModel as DBBaseModel
from ..utils.logger import get_logger


# =============================================================================
# 类型定义
# =============================================================================

ModelType = TypeVar("ModelType", bound=DBBaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


# =============================================================================
# 基础服务类
# =============================================================================

class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    服务层基础类
    
    提供通用的CRUD操作和业务逻辑基础功能
    """
    
    def __init__(self, db: AsyncSession, model: Type[ModelType]):
        """
        初始化基础服务
        
        Args:
            db: 数据库会话
            model: 数据模型类
        """
        self.db = db
        self.model = model
        self.cache = CacheManager()
        self.logger = get_logger(self.__class__.__name__)
    
    # =========================================================================
    # 基础CRUD操作
    # =========================================================================
    
    async def get_by_id(
        self,
        id: int,
        load_relationships: Optional[List[str]] = None
    ) -> Optional[ModelType]:
        """
        根据ID获取记录
        
        Args:
            id: 记录ID
            load_relationships: 需要加载的关联关系
            
        Returns:
            模型实例或None
        """
        try:
            query = select(self.model).where(self.model.id == id)
            
            # 加载关联关系
            if load_relationships:
                for relationship in load_relationships:
                    query = query.options(selectinload(getattr(self.model, relationship)))
            
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            self.logger.error(f"Error getting {self.model.__name__} by id {id}: {e}")
            raise InternalServerException(f"获取{self.model.__name__}失败")
    
    async def get_by_field(
        self,
        field_name: str,
        field_value: Any,
        load_relationships: Optional[List[str]] = None
    ) -> Optional[ModelType]:
        """
        根据字段获取记录
        
        Args:
            field_name: 字段名
            field_value: 字段值
            load_relationships: 需要加载的关联关系
            
        Returns:
            模型实例或None
        """
        try:
            field = getattr(self.model, field_name)
            query = select(self.model).where(field == field_value)
            
            # 加载关联关系
            if load_relationships:
                for relationship in load_relationships:
                    query = query.options(selectinload(getattr(self.model, relationship)))
            
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
            
        except AttributeError:
            raise ValidationException(f"字段 {field_name} 不存在")
        except Exception as e:
            self.logger.error(f"Error getting {self.model.__name__} by {field_name}: {e}")
            raise InternalServerException(f"获取{self.model.__name__}失败")
    
    async def get_all(
        self,
        filters: Optional[List] = None,
        order_by: Optional[str] = None,
        order_desc: bool = True,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        load_relationships: Optional[List[str]] = None
    ) -> List[ModelType]:
        """
        获取所有记录
        
        Args:
            filters: 过滤条件列表
            order_by: 排序字段
            order_desc: 是否降序
            limit: 限制数量
            offset: 偏移量
            load_relationships: 需要加载的关联关系
            
        Returns:
            模型实例列表
        """
        try:
            query = select(self.model)
            
            # 应用过滤条件
            if filters:
                query = query.where(and_(*filters))
            
            # 应用排序
            if order_by:
                order_field = getattr(self.model, order_by)
                if order_desc:
                    query = query.order_by(order_field.desc())
                else:
                    query = query.order_by(order_field)
            
            # 应用分页
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            
            # 加载关联关系
            if load_relationships:
                for relationship in load_relationships:
                    query = query.options(selectinload(getattr(self.model, relationship)))
            
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            self.logger.error(f"Error getting all {self.model.__name__}: {e}")
            raise InternalServerException(f"获取{self.model.__name__}列表失败")
    
    async def get_with_pagination(
        self,
        page: int = 1,
        size: int = 20,
        filters: Optional[List] = None,
        order_by: Optional[str] = None,
        order_desc: bool = True,
        load_relationships: Optional[List[str]] = None
    ) -> tuple[List[ModelType], int]:
        """
        分页获取记录
        
        Args:
            page: 页码
            size: 每页数量
            filters: 过滤条件列表
            order_by: 排序字段
            order_desc: 是否降序
            load_relationships: 需要加载的关联关系
            
        Returns:
            (记录列表, 总数)
        """
        try:
            # 构建基础查询
            query = select(self.model)
            count_query = select(func.count(self.model.id))
            
            # 应用过滤条件
            if filters:
                query = query.where(and_(*filters))
                count_query = count_query.where(and_(*filters))
            
            # 获取总数
            total_result = await self.db.execute(count_query)
            total = total_result.scalar()
            
            # 应用排序
            if order_by:
                order_field = getattr(self.model, order_by)
                if order_desc:
                    query = query.order_by(order_field.desc())
                else:
                    query = query.order_by(order_field)
            
            # 应用分页
            offset = (page - 1) * size
            query = query.offset(offset).limit(size)
            
            # 加载关联关系
            if load_relationships:
                for relationship in load_relationships:
                    query = query.options(selectinload(getattr(self.model, relationship)))
            
            result = await self.db.execute(query)
            items = result.scalars().all()
            
            return items, total
            
        except Exception as e:
            self.logger.error(f"Error getting paginated {self.model.__name__}: {e}")
            raise InternalServerException(f"获取{self.model.__name__}分页数据失败")
    
    async def create(
        self,
        obj_in: Union[CreateSchemaType, Dict[str, Any]],
        commit: bool = True
    ) -> ModelType:
        """
        创建记录
        
        Args:
            obj_in: 创建数据
            commit: 是否提交事务
            
        Returns:
            创建的模型实例
        """
        try:
            # 转换数据
            if isinstance(obj_in, dict):
                obj_data = obj_in
            else:
                obj_data = obj_in.dict(exclude_unset=True)
            
            # 创建实例
            db_obj = self.model(**obj_data)
            self.db.add(db_obj)
            
            if commit:
                await self.db.commit()
                await self.db.refresh(db_obj)
            
            return db_obj
            
        except Exception as e:
            if commit:
                await self.db.rollback()
            self.logger.error(f"Error creating {self.model.__name__}: {e}")
            raise InternalServerException(f"创建{self.model.__name__}失败")
    
    async def update(
        self,
        id: int,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]],
        commit: bool = True
    ) -> Optional[ModelType]:
        """
        更新记录
        
        Args:
            id: 记录ID
            obj_in: 更新数据
            commit: 是否提交事务
            
        Returns:
            更新后的模型实例
        """
        try:
            # 获取现有记录
            db_obj = await self.get_by_id(id)
            if not db_obj:
                raise ResourceNotFoundException(f"{self.model.__name__}不存在")
            
            # 转换数据
            if isinstance(obj_in, dict):
                update_data = obj_in
            else:
                update_data = obj_in.dict(exclude_unset=True)
            
            # 更新字段
            for field, value in update_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            if commit:
                await self.db.commit()
                await self.db.refresh(db_obj)
            
            return db_obj
            
        except ResourceNotFoundException:
            raise
        except Exception as e:
            if commit:
                await self.db.rollback()
            self.logger.error(f"Error updating {self.model.__name__} {id}: {e}")
            raise InternalServerException(f"更新{self.model.__name__}失败")
    
    async def delete(
        self,
        id: int,
        soft_delete: bool = True,
        commit: bool = True
    ) -> bool:
        """
        删除记录
        
        Args:
            id: 记录ID
            soft_delete: 是否软删除
            commit: 是否提交事务
            
        Returns:
            是否删除成功
        """
        try:
            # 获取现有记录
            db_obj = await self.get_by_id(id)
            if not db_obj:
                raise ResourceNotFoundException(f"{self.model.__name__}不存在")
            
            if soft_delete and hasattr(db_obj, 'is_deleted'):
                # 软删除
                db_obj.is_deleted = True
                if hasattr(db_obj, 'deleted_at'):
                    db_obj.deleted_at = datetime.utcnow()
            else:
                # 硬删除
                await self.db.delete(db_obj)
            
            if commit:
                await self.db.commit()
            
            return True
            
        except ResourceNotFoundException:
            raise
        except Exception as e:
            if commit:
                await self.db.rollback()
            self.logger.error(f"Error deleting {self.model.__name__} {id}: {e}")
            raise InternalServerException(f"删除{self.model.__name__}失败")
    
    async def bulk_create(
        self,
        objects: List[Union[CreateSchemaType, Dict[str, Any]]],
        commit: bool = True
    ) -> List[ModelType]:
        """
        批量创建记录
        
        Args:
            objects: 创建数据列表
            commit: 是否提交事务
            
        Returns:
            创建的模型实例列表
        """
        try:
            db_objects = []
            
            for obj_in in objects:
                # 转换数据
                if isinstance(obj_in, dict):
                    obj_data = obj_in
                else:
                    obj_data = obj_in.dict(exclude_unset=True)
                
                # 创建实例
                db_obj = self.model(**obj_data)
                db_objects.append(db_obj)
            
            self.db.add_all(db_objects)
            
            if commit:
                await self.db.commit()
                for db_obj in db_objects:
                    await self.db.refresh(db_obj)
            
            return db_objects
            
        except Exception as e:
            if commit:
                await self.db.rollback()
            self.logger.error(f"Error bulk creating {self.model.__name__}: {e}")
            raise InternalServerException(f"批量创建{self.model.__name__}失败")
    
    async def bulk_update(
        self,
        updates: List[Dict[str, Any]],
        commit: bool = True
    ) -> int:
        """
        批量更新记录
        
        Args:
            updates: 更新数据列表，每个字典必须包含id字段
            commit: 是否提交事务
            
        Returns:
            更新的记录数
        """
        try:
            updated_count = 0
            
            for update_data in updates:
                if 'id' not in update_data:
                    continue
                
                record_id = update_data.pop('id')
                
                stmt = (
                    update(self.model)
                    .where(self.model.id == record_id)
                    .values(**update_data)
                )
                
                result = await self.db.execute(stmt)
                updated_count += result.rowcount
            
            if commit:
                await self.db.commit()
            
            return updated_count
            
        except Exception as e:
            if commit:
                await self.db.rollback()
            self.logger.error(f"Error bulk updating {self.model.__name__}: {e}")
            raise InternalServerException(f"批量更新{self.model.__name__}失败")
    
    async def bulk_delete(
        self,
        ids: List[int],
        soft_delete: bool = True,
        commit: bool = True
    ) -> int:
        """
        批量删除记录
        
        Args:
            ids: 记录ID列表
            soft_delete: 是否软删除
            commit: 是否提交事务
            
        Returns:
            删除的记录数
        """
        try:
            if soft_delete and hasattr(self.model, 'is_deleted'):
                # 软删除
                update_data = {'is_deleted': True}
                if hasattr(self.model, 'deleted_at'):
                    update_data['deleted_at'] = datetime.utcnow()
                
                stmt = (
                    update(self.model)
                    .where(self.model.id.in_(ids))
                    .values(**update_data)
                )
                
                result = await self.db.execute(stmt)
                deleted_count = result.rowcount
            else:
                # 硬删除
                stmt = delete(self.model).where(self.model.id.in_(ids))
                result = await self.db.execute(stmt)
                deleted_count = result.rowcount
            
            if commit:
                await self.db.commit()
            
            return deleted_count
            
        except Exception as e:
            if commit:
                await self.db.rollback()
            self.logger.error(f"Error bulk deleting {self.model.__name__}: {e}")
            raise InternalServerException(f"批量删除{self.model.__name__}失败")
    
    # =========================================================================
    # 缓存操作
    # =========================================================================
    
    def _get_cache_key(self, key: str) -> str:
        """
        生成缓存键
        
        Args:
            key: 基础键
            
        Returns:
            完整的缓存键
        """
        return f"{self.model.__name__.lower()}:{key}"
    
    async def get_from_cache(self, key: str) -> Any:
        """
        从缓存获取数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存数据
        """
        cache_key = self._get_cache_key(key)
        return await self.cache.get(cache_key)
    
    async def set_to_cache(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒）
            
        Returns:
            是否设置成功
        """
        cache_key = self._get_cache_key(key)
        return await self.cache.set(cache_key, value, expire=expire)
    
    async def delete_from_cache(self, key: str) -> bool:
        """
        删除缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        cache_key = self._get_cache_key(key)
        return await self.cache.delete(cache_key)
    
    async def clear_cache_pattern(self, pattern: str) -> int:
        """
        清理匹配模式的缓存
        
        Args:
            pattern: 缓存键模式
            
        Returns:
            清理的缓存数量
        """
        cache_pattern = self._get_cache_key(pattern)
        return await self.cache.clear_pattern(cache_pattern)
    
    # =========================================================================
    # 事务管理
    # =========================================================================
    
    async def begin_transaction(self):
        """
        开始事务
        """
        await self.db.begin()
    
    async def commit_transaction(self):
        """
        提交事务
        """
        await self.db.commit()
    
    async def rollback_transaction(self):
        """
        回滚事务
        """
        await self.db.rollback()
    
    # =========================================================================
    # 统计和聚合
    # =========================================================================
    
    async def count(
        self,
        filters: Optional[List] = None
    ) -> int:
        """
        统计记录数量
        
        Args:
            filters: 过滤条件列表
            
        Returns:
            记录数量
        """
        try:
            query = select(func.count(self.model.id))
            
            if filters:
                query = query.where(and_(*filters))
            
            result = await self.db.execute(query)
            return result.scalar()
            
        except Exception as e:
            self.logger.error(f"Error counting {self.model.__name__}: {e}")
            raise InternalServerException(f"统计{self.model.__name__}数量失败")
    
    async def exists(
        self,
        filters: List
    ) -> bool:
        """
        检查记录是否存在
        
        Args:
            filters: 过滤条件列表
            
        Returns:
            是否存在
        """
        try:
            query = select(self.model.id).where(and_(*filters)).limit(1)
            result = await self.db.execute(query)
            return result.scalar_one_or_none() is not None
            
        except Exception as e:
            self.logger.error(f"Error checking existence of {self.model.__name__}: {e}")
            raise InternalServerException(f"检查{self.model.__name__}存在性失败")
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证数据
        
        Args:
            data: 待验证数据
            
        Returns:
            验证后的数据
        """
        # 子类可以重写此方法实现自定义验证
        return data
    
    def log_operation(
        self,
        operation: str,
        record_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        记录操作日志
        
        Args:
            operation: 操作类型
            record_id: 记录ID
            details: 操作详情
        """
        log_data = {
            "model": self.model.__name__,
            "operation": operation,
            "record_id": record_id,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"Operation logged: {log_data}")