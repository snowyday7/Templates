#!/usr/bin/env python3
"""
管理员服务

提供系统管理相关的业务逻辑，包括：
1. 角色权限管理
2. 系统监控
3. 用户管理
4. 系统配置
5. 日志管理
6. 数据统计
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, text
from sqlalchemy.orm import selectinload, joinedload

from .base import BaseService
from .user import UserService
from ..core.exceptions import (
    ValidationException,
    ResourceNotFoundException,
    ConflictException,
    AuthorizationException
)
from ..models.user import (
    User, UserRole, UserPermission, UserSession,
    user_roles, role_permissions, user_permissions
)
from ..models.auth import LoginAttempt, AuthToken
from ..utils.logger import get_logger
from ..core.config import get_settings


# =============================================================================
# 管理员服务类
# =============================================================================

class AdminService(BaseService[User, dict, dict]):
    """
    管理员服务类
    
    提供系统管理相关的业务逻辑操作
    """
    
    def __init__(self, db: AsyncSession):
        """
        初始化管理员服务
        
        Args:
            db: 数据库会话
        """
        super().__init__(db, User)
        self.logger = get_logger(__name__)
        self.user_service = UserService(db)
        self.settings = get_settings()
    
    # =========================================================================
    # 角色管理
    # =========================================================================
    
    async def create_role(
        self,
        name: str,
        display_name: str,
        description: Optional[str] = None,
        permissions: Optional[List[int]] = None,
        created_by: Optional[int] = None
    ) -> UserRole:
        """
        创建角色
        
        Args:
            name: 角色名称
            display_name: 显示名称
            description: 角色描述
            permissions: 权限ID列表
            created_by: 创建者ID
            
        Returns:
            创建的角色对象
        """
        try:
            # 检查角色名是否已存在
            existing_role = await self._get_role_by_name(name)
            if existing_role:
                raise ConflictException("角色名已存在")
            
            # 创建角色
            role_data = {
                "name": name,
                "display_name": display_name,
                "description": description,
                "is_active": True,
                "created_by": created_by,
                "created_at": datetime.utcnow()
            }
            
            role = UserRole(**role_data)
            self.db.add(role)
            await self.db.flush()  # 获取角色ID
            
            # 分配权限
            if permissions:
                await self._assign_permissions_to_role(role.id, permissions)
            
            await self.db.commit()
            
            # 记录操作日志
            self.log_operation("create_role", role.id, {
                "name": name,
                "permissions": permissions,
                "created_by": created_by
            })
            
            return role
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error creating role {name}: {e}")
            raise
    
    async def update_role(
        self,
        role_id: int,
        update_data: Dict[str, Any],
        updated_by: Optional[int] = None
    ) -> UserRole:
        """
        更新角色
        
        Args:
            role_id: 角色ID
            update_data: 更新数据
            updated_by: 更新者ID
            
        Returns:
            更新后的角色对象
        """
        try:
            # 获取角色
            role = await self._get_role_by_id(role_id)
            if not role:
                raise ResourceNotFoundException("角色不存在")
            
            # 检查角色名冲突
            if "name" in update_data and update_data["name"] != role.name:
                existing_role = await self._get_role_by_name(update_data["name"])
                if existing_role and existing_role.id != role_id:
                    raise ConflictException("角色名已存在")
            
            # 更新基本信息
            permissions = update_data.pop("permissions", None)
            update_data["updated_by"] = updated_by
            update_data["updated_at"] = datetime.utcnow()
            
            for key, value in update_data.items():
                if hasattr(role, key):
                    setattr(role, key, value)
            
            # 更新权限
            if permissions is not None:
                await self._update_role_permissions(role_id, permissions)
            
            await self.db.commit()
            
            # 记录操作日志
            self.log_operation("update_role", role_id, {
                "fields": list(update_data.keys()),
                "updated_by": updated_by
            })
            
            return role
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error updating role {role_id}: {e}")
            raise
    
    async def delete_role(
        self,
        role_id: int,
        deleted_by: Optional[int] = None
    ) -> bool:
        """
        删除角色
        
        Args:
            role_id: 角色ID
            deleted_by: 删除者ID
            
        Returns:
            是否删除成功
        """
        try:
            # 获取角色
            role = await self._get_role_by_id(role_id)
            if not role:
                raise ResourceNotFoundException("角色不存在")
            
            # 检查是否有用户使用该角色
            user_count = await self._count_users_with_role(role_id)
            if user_count > 0:
                raise ConflictException(f"该角色正在被 {user_count} 个用户使用，无法删除")
            
            # 删除角色权限关联
            await self._remove_all_role_permissions(role_id)
            
            # 删除角色
            await self.db.delete(role)
            await self.db.commit()
            
            # 记录操作日志
            self.log_operation("delete_role", role_id, {
                "role_name": role.name,
                "deleted_by": deleted_by
            })
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error deleting role {role_id}: {e}")
            raise
    
    async def get_roles(
        self,
        page: int = 1,
        size: int = 20,
        search: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> tuple[List[UserRole], int]:
        """
        获取角色列表
        
        Args:
            page: 页码
            size: 每页数量
            search: 搜索关键词
            is_active: 是否活跃
            
        Returns:
            (角色列表, 总数)
        """
        try:
            query = select(UserRole).options(
                selectinload(UserRole.permissions)
            )
            
            # 添加过滤条件
            filters = []
            if search:
                filters.append(
                    or_(
                        UserRole.name.ilike(f"%{search}%"),
                        UserRole.display_name.ilike(f"%{search}%"),
                        UserRole.description.ilike(f"%{search}%")
                    )
                )
            
            if is_active is not None:
                filters.append(UserRole.is_active == is_active)
            
            if filters:
                query = query.where(and_(*filters))
            
            # 计算总数
            count_query = select(func.count(UserRole.id))
            if filters:
                count_query = count_query.where(and_(*filters))
            
            count_result = await self.db.execute(count_query)
            total = count_result.scalar()
            
            # 分页查询
            query = query.order_by(UserRole.created_at.desc())
            query = query.offset((page - 1) * size).limit(size)
            
            result = await self.db.execute(query)
            roles = result.scalars().all()
            
            return roles, total
            
        except Exception as e:
            self.logger.error(f"Error getting roles: {e}")
            raise
    
    async def get_role_permissions(self, role_id: int) -> List[UserPermission]:
        """
        获取角色权限
        
        Args:
            role_id: 角色ID
            
        Returns:
            权限列表
        """
        try:
            query = (
                select(UserPermission)
                .join(role_permissions)
                .where(role_permissions.c.role_id == role_id)
                .where(UserPermission.is_active == True)
                .order_by(UserPermission.category, UserPermission.name)
            )
            
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            self.logger.error(f"Error getting role permissions for role {role_id}: {e}")
            raise
    
    # =========================================================================
    # 权限管理
    # =========================================================================
    
    async def create_permission(
        self,
        name: str,
        display_name: str,
        description: Optional[str] = None,
        category: str = "general",
        created_by: Optional[int] = None
    ) -> UserPermission:
        """
        创建权限
        
        Args:
            name: 权限名称
            display_name: 显示名称
            description: 权限描述
            category: 权限分类
            created_by: 创建者ID
            
        Returns:
            创建的权限对象
        """
        try:
            # 检查权限名是否已存在
            existing_permission = await self._get_permission_by_name(name)
            if existing_permission:
                raise ConflictException("权限名已存在")
            
            # 创建权限
            permission_data = {
                "name": name,
                "display_name": display_name,
                "description": description,
                "category": category,
                "is_active": True,
                "created_by": created_by,
                "created_at": datetime.utcnow()
            }
            
            permission = UserPermission(**permission_data)
            self.db.add(permission)
            await self.db.commit()
            
            # 记录操作日志
            self.log_operation("create_permission", permission.id, {
                "name": name,
                "category": category,
                "created_by": created_by
            })
            
            return permission
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error creating permission {name}: {e}")
            raise
    
    async def get_permissions(
        self,
        category: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> List[UserPermission]:
        """
        获取权限列表
        
        Args:
            category: 权限分类
            is_active: 是否活跃
            
        Returns:
            权限列表
        """
        try:
            query = select(UserPermission)
            
            filters = []
            if category:
                filters.append(UserPermission.category == category)
            
            if is_active is not None:
                filters.append(UserPermission.is_active == is_active)
            
            if filters:
                query = query.where(and_(*filters))
            
            query = query.order_by(UserPermission.category, UserPermission.name)
            
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            self.logger.error(f"Error getting permissions: {e}")
            raise
    
    async def get_permission_categories(self) -> List[str]:
        """
        获取权限分类列表
        
        Returns:
            分类列表
        """
        try:
            query = (
                select(UserPermission.category)
                .distinct()
                .where(UserPermission.is_active == True)
                .order_by(UserPermission.category)
            )
            
            result = await self.db.execute(query)
            return [row[0] for row in result.fetchall()]
            
        except Exception as e:
            self.logger.error(f"Error getting permission categories: {e}")
            raise
    
    # =========================================================================
    # 用户管理
    # =========================================================================
    
    async def get_users_with_details(
        self,
        page: int = 1,
        size: int = 20,
        search: Optional[str] = None,
        status: Optional[str] = None,
        role_id: Optional[int] = None,
        is_active: Optional[bool] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        获取用户详细信息列表
        
        Args:
            page: 页码
            size: 每页数量
            search: 搜索关键词
            status: 用户状态
            role_id: 角色ID
            is_active: 是否活跃
            
        Returns:
            (用户列表, 总数)
        """
        try:
            # 构建查询
            query = (
                select(User)
                .options(
                    selectinload(User.roles),
                    selectinload(User.sessions)
                )
            )
            
            # 添加过滤条件
            filters = []
            
            if search:
                filters.append(
                    or_(
                        User.username.ilike(f"%{search}%"),
                        User.email.ilike(f"%{search}%"),
                        User.first_name.ilike(f"%{search}%"),
                        User.last_name.ilike(f"%{search}%"),
                        User.display_name.ilike(f"%{search}%")
                    )
                )
            
            if status:
                filters.append(User.status == status)
            
            if is_active is not None:
                filters.append(User.is_active == is_active)
            
            if role_id:
                filters.append(
                    User.id.in_(
                        select(user_roles.c.user_id)
                        .where(user_roles.c.role_id == role_id)
                    )
                )
            
            if filters:
                query = query.where(and_(*filters))
            
            # 计算总数
            count_query = select(func.count(User.id))
            if filters:
                count_query = count_query.where(and_(*filters))
            
            count_result = await self.db.execute(count_query)
            total = count_result.scalar()
            
            # 分页查询
            query = query.order_by(User.created_at.desc())
            query = query.offset((page - 1) * size).limit(size)
            
            result = await self.db.execute(query)
            users = result.scalars().all()
            
            # 构建详细信息
            user_details = []
            for user in users:
                # 获取活跃会话数
                active_sessions = len([
                    s for s in user.sessions 
                    if s.status == 'active' and s.expires_at > datetime.utcnow()
                ])
                
                user_details.append({
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "display_name": user.display_name,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "avatar_url": user.avatar_url,
                    "status": user.status,
                    "is_active": user.is_active,
                    "is_superuser": user.is_superuser,
                    "email_verified_at": user.email_verified_at,
                    "last_login_at": user.last_login_at,
                    "last_activity_at": user.last_activity_at,
                    "created_at": user.created_at,
                    "roles": [{
                        "id": role.id,
                        "name": role.name,
                        "display_name": role.display_name
                    } for role in user.roles],
                    "active_sessions": active_sessions,
                    "failed_login_attempts": user.failed_login_attempts,
                    "locked_until": user.locked_until
                })
            
            return user_details, total
            
        except Exception as e:
            self.logger.error(f"Error getting users with details: {e}")
            raise
    
    async def assign_user_role(
        self,
        user_id: int,
        role_id: int,
        assigned_by: int,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        为用户分配角色
        
        Args:
            user_id: 用户ID
            role_id: 角色ID
            assigned_by: 分配者ID
            expires_at: 过期时间
            
        Returns:
            是否分配成功
        """
        return await self.user_service.assign_role(
            user_id, role_id, assigned_by, expires_at
        )
    
    async def remove_user_role(
        self,
        user_id: int,
        role_id: int,
        removed_by: int
    ) -> bool:
        """
        移除用户角色
        
        Args:
            user_id: 用户ID
            role_id: 角色ID
            removed_by: 移除者ID
            
        Returns:
            是否移除成功
        """
        return await self.user_service.remove_role(
            user_id, role_id, removed_by
        )
    
    # =========================================================================
    # 系统监控
    # =========================================================================
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            系统统计数据
        """
        try:
            # 用户统计
            user_stats = await self.user_service.get_user_statistics()
            
            # 登录统计
            login_stats = await self._get_login_statistics()
            
            # 会话统计
            session_stats = await self._get_session_statistics()
            
            # 令牌统计
            token_stats = await self._get_token_statistics()
            
            # 系统资源统计
            resource_stats = await self._get_resource_statistics()
            
            return {
                "users": user_stats,
                "logins": login_stats,
                "sessions": session_stats,
                "tokens": token_stats,
                "resources": resource_stats,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system statistics: {e}")
            raise
    
    async def get_login_attempts(
        self,
        page: int = 1,
        size: int = 50,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        result: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> tuple[List[LoginAttempt], int]:
        """
        获取登录尝试记录
        
        Args:
            page: 页码
            size: 每页数量
            user_id: 用户ID
            ip_address: IP地址
            result: 登录结果
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            (登录尝试列表, 总数)
        """
        try:
            query = select(LoginAttempt)
            
            # 添加过滤条件
            filters = []
            
            if user_id:
                filters.append(LoginAttempt.user_id == user_id)
            
            if ip_address:
                filters.append(LoginAttempt.ip_address == ip_address)
            
            if result:
                filters.append(LoginAttempt.result == result)
            
            if start_date:
                filters.append(LoginAttempt.created_at >= start_date)
            
            if end_date:
                filters.append(LoginAttempt.created_at <= end_date)
            
            if filters:
                query = query.where(and_(*filters))
            
            # 计算总数
            count_query = select(func.count(LoginAttempt.id))
            if filters:
                count_query = count_query.where(and_(*filters))
            
            count_result = await self.db.execute(count_query)
            total = count_result.scalar()
            
            # 分页查询
            query = query.order_by(LoginAttempt.created_at.desc())
            query = query.offset((page - 1) * size).limit(size)
            
            result = await self.db.execute(query)
            attempts = result.scalars().all()
            
            return attempts, total
            
        except Exception as e:
            self.logger.error(f"Error getting login attempts: {e}")
            raise
    
    async def get_suspicious_activities(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        获取可疑活动
        
        Args:
            days: 查询天数
            
        Returns:
            可疑活动列表
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # 多次失败登录的IP
            failed_login_query = text("""
                SELECT 
                    ip_address,
                    COUNT(*) as attempt_count,
                    COUNT(DISTINCT user_id) as user_count,
                    MAX(created_at) as last_attempt
                FROM login_attempts 
                WHERE result = 'failed' 
                    AND created_at >= :start_date
                    AND ip_address IS NOT NULL
                GROUP BY ip_address 
                HAVING COUNT(*) >= 10
                ORDER BY attempt_count DESC
            ")
            
            result = await self.db.execute(failed_login_query, {"start_date": start_date})
            failed_logins = result.fetchall()
            
            # 异常登录时间的用户
            unusual_time_query = text("""
                SELECT 
                    user_id,
                    COUNT(*) as login_count,
                    MIN(created_at) as first_login,
                    MAX(created_at) as last_login
                FROM login_attempts 
                WHERE result = 'success' 
                    AND created_at >= :start_date
                    AND (EXTRACT(HOUR FROM created_at) < 6 OR EXTRACT(HOUR FROM created_at) > 23)
                GROUP BY user_id 
                HAVING COUNT(*) >= 5
                ORDER BY login_count DESC
            ")
            
            result = await self.db.execute(unusual_time_query, {"start_date": start_date})
            unusual_times = result.fetchall()
            
            # 多地登录的用户（基于IP地址变化）
            multiple_locations_query = text("""
                SELECT 
                    user_id,
                    COUNT(DISTINCT ip_address) as ip_count,
                    COUNT(*) as login_count
                FROM login_attempts 
                WHERE result = 'success' 
                    AND created_at >= :start_date
                    AND user_id IS NOT NULL
                    AND ip_address IS NOT NULL
                GROUP BY user_id 
                HAVING COUNT(DISTINCT ip_address) >= 5
                ORDER BY ip_count DESC
            ")
            
            result = await self.db.execute(multiple_locations_query, {"start_date": start_date})
            multiple_locations = result.fetchall()
            
            activities = []
            
            # 处理失败登录
            for row in failed_logins:
                activities.append({
                    "type": "multiple_failed_logins",
                    "severity": "high" if row.attempt_count >= 50 else "medium",
                    "ip_address": row.ip_address,
                    "attempt_count": row.attempt_count,
                    "user_count": row.user_count,
                    "last_attempt": row.last_attempt,
                    "description": f"IP {row.ip_address} 在 {days} 天内失败登录 {row.attempt_count} 次"
                })
            
            # 处理异常时间登录
            for row in unusual_times:
                activities.append({
                    "type": "unusual_login_time",
                    "severity": "medium",
                    "user_id": row.user_id,
                    "login_count": row.login_count,
                    "first_login": row.first_login,
                    "last_login": row.last_login,
                    "description": f"用户 {row.user_id} 在异常时间段登录 {row.login_count} 次"
                })
            
            # 处理多地登录
            for row in multiple_locations:
                activities.append({
                    "type": "multiple_locations",
                    "severity": "medium",
                    "user_id": row.user_id,
                    "ip_count": row.ip_count,
                    "login_count": row.login_count,
                    "description": f"用户 {row.user_id} 从 {row.ip_count} 个不同IP地址登录"
                })
            
            return activities
            
        except Exception as e:
            self.logger.error(f"Error getting suspicious activities: {e}")
            raise
    
    # =========================================================================
    # 系统配置
    # =========================================================================
    
    async def get_system_config(self) -> Dict[str, Any]:
        """
        获取系统配置
        
        Returns:
            系统配置信息
        """
        try:
            # 从缓存获取配置
            config = await self.cache.get("system:config")
            if config:
                return config
            
            # 构建配置信息
            config = {
                "app_name": self.settings.app_name,
                "app_version": self.settings.app_version,
                "environment": self.settings.environment,
                "debug": self.settings.debug,
                "database_url": self.settings.database_url.replace(
                    self.settings.database_url.split("@")[0].split("//")[1],
                    "***:***"
                ) if "@" in self.settings.database_url else "***",
                "redis_url": "***" if self.settings.redis_url else None,
                "email_enabled": bool(self.settings.smtp_host),
                "file_upload_max_size": self.settings.max_file_size,
                "allowed_file_types": self.settings.allowed_file_types,
                "jwt_expire_minutes": self.settings.access_token_expire_minutes,
                "password_min_length": 8,
                "max_login_attempts": 5,
                "lockout_duration_minutes": 30,
                "session_timeout_hours": 24,
                "features": {
                    "user_registration": True,
                    "email_verification": True,
                    "password_reset": True,
                    "two_factor_auth": True,
                    "file_upload": True,
                    "api_rate_limiting": True
                }
            }
            
            # 缓存配置（5分钟）
            await self.cache.set("system:config", config, expire=300)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error getting system config: {e}")
            raise
    
    async def update_system_config(
        self,
        config_updates: Dict[str, Any],
        updated_by: int
    ) -> Dict[str, Any]:
        """
        更新系统配置
        
        Args:
            config_updates: 配置更新
            updated_by: 更新者ID
            
        Returns:
            更新后的配置
        """
        try:
            # 验证配置更新
            allowed_updates = {
                "password_min_length",
                "max_login_attempts",
                "lockout_duration_minutes",
                "session_timeout_hours",
                "features"
            }
            
            invalid_keys = set(config_updates.keys()) - allowed_updates
            if invalid_keys:
                raise ValidationException(f"不允许更新的配置项: {invalid_keys}")
            
            # 获取当前配置
            current_config = await self.get_system_config()
            
            # 更新配置
            for key, value in config_updates.items():
                if key == "features" and isinstance(value, dict):
                    current_config["features"].update(value)
                else:
                    current_config[key] = value
            
            # 保存到缓存
            await self.cache.set("system:config", current_config, expire=300)
            
            # 记录操作日志
            self.log_operation("update_system_config", updated_by, {
                "updates": config_updates,
                "updated_by": updated_by
            })
            
            return current_config
            
        except Exception as e:
            self.logger.error(f"Error updating system config: {e}")
            raise
    
    # =========================================================================
    # 系统维护
    # =========================================================================
    
    async def cleanup_expired_data(self) -> Dict[str, int]:
        """
        清理过期数据
        
        Returns:
            清理统计信息
        """
        try:
            cleanup_stats = {}
            
            # 清理过期令牌
            expired_tokens = await self.db.execute(
                delete(AuthToken).where(
                    AuthToken.expires_at < datetime.utcnow()
                )
            )
            cleanup_stats["expired_tokens"] = expired_tokens.rowcount
            
            # 清理过期会话
            expired_sessions = await self.db.execute(
                delete(UserSession).where(
                    UserSession.expires_at < datetime.utcnow()
                )
            )
            cleanup_stats["expired_sessions"] = expired_sessions.rowcount
            
            # 清理旧的登录尝试记录（保留30天）
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            old_login_attempts = await self.db.execute(
                delete(LoginAttempt).where(
                    LoginAttempt.created_at < thirty_days_ago
                )
            )
            cleanup_stats["old_login_attempts"] = old_login_attempts.rowcount
            
            await self.db.commit()
            
            # 记录操作日志
            self.log_operation("cleanup_expired_data", None, cleanup_stats)
            
            return cleanup_stats
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error cleaning up expired data: {e}")
            raise
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            缓存信息
        """
        try:
            # 获取Redis信息
            info = await self.cache.redis.info()
            
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")
            return {}
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        清理缓存
        
        Args:
            pattern: 缓存键模式
            
        Returns:
            清理的键数量
        """
        try:
            if pattern:
                return await self.cache.clear_pattern(pattern)
            else:
                return await self.cache.clear_all()
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return 0
    
    # =========================================================================
    # 私有辅助方法
    # =========================================================================
    
    async def _get_role_by_id(self, role_id: int) -> Optional[UserRole]:
        """根据ID获取角色"""
        try:
            query = select(UserRole).where(UserRole.id == role_id)
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
        except Exception:
            return None
    
    async def _get_role_by_name(self, name: str) -> Optional[UserRole]:
        """根据名称获取角色"""
        try:
            query = select(UserRole).where(UserRole.name == name)
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
        except Exception:
            return None
    
    async def _get_permission_by_name(self, name: str) -> Optional[UserPermission]:
        """根据名称获取权限"""
        try:
            query = select(UserPermission).where(UserPermission.name == name)
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
        except Exception:
            return None
    
    async def _assign_permissions_to_role(self, role_id: int, permission_ids: List[int]):
        """为角色分配权限"""
        try:
            for permission_id in permission_ids:
                insert_stmt = role_permissions.insert().values(
                    role_id=role_id,
                    permission_id=permission_id
                )
                await self.db.execute(insert_stmt)
        except Exception as e:
            self.logger.error(f"Error assigning permissions to role: {e}")
            raise
    
    async def _update_role_permissions(self, role_id: int, permission_ids: List[int]):
        """更新角色权限"""
        try:
            # 删除现有权限
            await self._remove_all_role_permissions(role_id)
            
            # 添加新权限
            if permission_ids:
                await self._assign_permissions_to_role(role_id, permission_ids)
                
        except Exception as e:
            self.logger.error(f"Error updating role permissions: {e}")
            raise
    
    async def _remove_all_role_permissions(self, role_id: int):
        """移除角色所有权限"""
        try:
            delete_stmt = delete(role_permissions).where(
                role_permissions.c.role_id == role_id
            )
            await self.db.execute(delete_stmt)
        except Exception as e:
            self.logger.error(f"Error removing all role permissions: {e}")
            raise
    
    async def _count_users_with_role(self, role_id: int) -> int:
        """统计使用该角色的用户数量"""
        try:
            query = select(func.count(user_roles.c.user_id)).where(
                user_roles.c.role_id == role_id
            )
            result = await self.db.execute(query)
            return result.scalar()
        except Exception:
            return 0
    
    async def _get_login_statistics(self) -> Dict[str, Any]:
        """获取登录统计"""
        try:
            # 今日登录
            today = datetime.utcnow().date()
            today_start = datetime.combine(today, datetime.min.time())
            
            today_logins = await self.db.execute(
                select(func.count(LoginAttempt.id))
                .where(LoginAttempt.result == "success")
                .where(LoginAttempt.created_at >= today_start)
            )
            
            # 本周登录
            week_start = today_start - timedelta(days=today.weekday())
            week_logins = await self.db.execute(
                select(func.count(LoginAttempt.id))
                .where(LoginAttempt.result == "success")
                .where(LoginAttempt.created_at >= week_start)
            )
            
            # 失败登录
            failed_logins = await self.db.execute(
                select(func.count(LoginAttempt.id))
                .where(LoginAttempt.result == "failed")
                .where(LoginAttempt.created_at >= today_start)
            )
            
            return {
                "today_logins": today_logins.scalar(),
                "week_logins": week_logins.scalar(),
                "today_failed_logins": failed_logins.scalar()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting login statistics: {e}")
            return {}
    
    async def _get_session_statistics(self) -> Dict[str, Any]:
        """获取会话统计"""
        try:
            # 活跃会话
            active_sessions = await self.db.execute(
                select(func.count(UserSession.id))
                .where(UserSession.status == "active")
                .where(UserSession.expires_at > datetime.utcnow())
            )
            
            # 总会话数
            total_sessions = await self.db.execute(
                select(func.count(UserSession.id))
            )
            
            return {
                "active_sessions": active_sessions.scalar(),
                "total_sessions": total_sessions.scalar()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session statistics: {e}")
            return {}
    
    async def _get_token_statistics(self) -> Dict[str, Any]:
        """获取令牌统计"""
        try:
            # 活跃令牌
            active_tokens = await self.db.execute(
                select(func.count(AuthToken.id))
                .where(AuthToken.is_revoked == False)
                .where(AuthToken.expires_at > datetime.utcnow())
            )
            
            # 已撤销令牌
            revoked_tokens = await self.db.execute(
                select(func.count(AuthToken.id))
                .where(AuthToken.is_revoked == True)
            )
            
            return {
                "active_tokens": active_tokens.scalar(),
                "revoked_tokens": revoked_tokens.scalar()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting token statistics: {e}")
            return {}
    
    async def _get_resource_statistics(self) -> Dict[str, Any]:
        """获取资源统计"""
        try:
            # 数据库连接数
            db_connections = await self.db.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            )
            
            # 数据库大小
            db_size = await self.db.execute(
                text("SELECT pg_size_pretty(pg_database_size(current_database()))")
            )
            
            return {
                "database_connections": db_connections.scalar(),
                "database_size": db_size.scalar()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting resource statistics: {e}")
            return {}