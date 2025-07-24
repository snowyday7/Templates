#!/usr/bin/env python3
"""
用户服务

提供用户相关的业务逻辑，包括：
1. 用户CRUD操作
2. 用户权限管理
3. 用户角色管理
4. 用户会话管理
5. 用户统计分析
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload, joinedload
from pydantic import BaseModel

from .base import BaseService
from ..core.security import get_password_hash, verify_password
from ..core.exceptions import (
    ValidationException,
    ResourceNotFoundException,
    ConflictException,
    AuthorizationException
)
from ..models.user import (
    User, UserRole, UserPermission, UserSession,
    UserStatus, UserGender, SessionStatus,
    user_roles, role_permissions, user_permissions
)
from ..utils.logger import get_logger


# =============================================================================
# 用户服务类
# =============================================================================

class UserService(BaseService[User, dict, dict]):
    """
    用户服务类
    
    提供用户相关的业务逻辑操作
    """
    
    def __init__(self, db: AsyncSession):
        """
        初始化用户服务
        
        Args:
            db: 数据库会话
        """
        super().__init__(db, User)
        self.logger = get_logger(__name__)
    
    # =========================================================================
    # 用户基础操作
    # =========================================================================
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        **kwargs
    ) -> User:
        """
        创建用户
        
        Args:
            username: 用户名
            email: 邮箱
            password: 密码
            **kwargs: 其他用户属性
            
        Returns:
            创建的用户对象
        """
        # 检查用户名是否已存在
        existing_user = await self.get_by_username(username)
        if existing_user:
            raise ConflictException("用户名已存在")
        
        # 检查邮箱是否已存在
        existing_email = await self.get_by_email(email)
        if existing_email:
            raise ConflictException("邮箱已存在")
        
        # 创建用户数据
        user_data = {
            "username": username,
            "email": email,
            "password_hash": get_password_hash(password),
            "status": UserStatus.ACTIVE,
            "is_active": True,
            "created_at": datetime.utcnow(),
            **kwargs
        }
        
        # 验证数据
        user_data = self.validate_user_data(user_data)
        
        # 创建用户
        user = await self.create(user_data)
        
        # 记录操作日志
        self.log_operation("create_user", user.id, {"username": username, "email": email})
        
        return user
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """
        根据用户名获取用户
        
        Args:
            username: 用户名
            
        Returns:
            用户对象或None
        """
        return await self.get_by_field("username", username)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """
        根据邮箱获取用户
        
        Args:
            email: 邮箱
            
        Returns:
            用户对象或None
        """
        return await self.get_by_field("email", email)
    
    async def get_by_phone(self, phone: str) -> Optional[User]:
        """
        根据手机号获取用户
        
        Args:
            phone: 手机号
            
        Returns:
            用户对象或None
        """
        return await self.get_by_field("phone", phone)
    
    async def update_user(
        self,
        user_id: int,
        update_data: Dict[str, Any],
        updated_by: Optional[int] = None
    ) -> User:
        """
        更新用户信息
        
        Args:
            user_id: 用户ID
            update_data: 更新数据
            updated_by: 更新者ID
            
        Returns:
            更新后的用户对象
        """
        # 获取用户
        user = await self.get_by_id(user_id)
        if not user:
            raise ResourceNotFoundException("用户不存在")
        
        # 检查敏感字段
        if "username" in update_data and update_data["username"] != user.username:
            existing_user = await self.get_by_username(update_data["username"])
            if existing_user and existing_user.id != user_id:
                raise ConflictException("用户名已存在")
        
        if "email" in update_data and update_data["email"] != user.email:
            existing_email = await self.get_by_email(update_data["email"])
            if existing_email and existing_email.id != user_id:
                raise ConflictException("邮箱已存在")
        
        if "phone" in update_data and update_data["phone"] != user.phone:
            existing_phone = await self.get_by_phone(update_data["phone"])
            if existing_phone and existing_phone.id != user_id:
                raise ConflictException("手机号已存在")
        
        # 处理密码更新
        if "password" in update_data:
            update_data["password_hash"] = get_password_hash(update_data.pop("password"))
            update_data["password_changed_at"] = datetime.utcnow()
        
        # 添加更新信息
        update_data["updated_at"] = datetime.utcnow()
        if updated_by:
            update_data["updated_by"] = updated_by
        
        # 验证数据
        update_data = self.validate_user_data(update_data, is_update=True)
        
        # 更新用户
        updated_user = await self.update(user_id, update_data)
        
        # 清理缓存
        await self.clear_user_cache(user_id)
        
        # 记录操作日志
        self.log_operation("update_user", user_id, {"fields": list(update_data.keys())})
        
        return updated_user
    
    async def change_password(
        self,
        user_id: int,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        修改用户密码
        
        Args:
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码
            
        Returns:
            是否修改成功
        """
        # 获取用户
        user = await self.get_by_id(user_id)
        if not user:
            raise ResourceNotFoundException("用户不存在")
        
        # 验证旧密码
        if not verify_password(old_password, user.password_hash):
            raise ValidationException("旧密码错误")
        
        # 更新密码
        await self.update_user(user_id, {
            "password": new_password,
            "failed_login_attempts": 0,  # 重置失败次数
            "locked_until": None  # 解除锁定
        })
        
        # 撤销所有会话（强制重新登录）
        await self.revoke_all_sessions(user_id)
        
        # 记录操作日志
        self.log_operation("change_password", user_id)
        
        return True
    
    async def reset_password(
        self,
        user_id: int,
        new_password: str,
        reset_by: Optional[int] = None
    ) -> bool:
        """
        重置用户密码（管理员操作）
        
        Args:
            user_id: 用户ID
            new_password: 新密码
            reset_by: 重置者ID
            
        Returns:
            是否重置成功
        """
        # 更新密码
        await self.update_user(user_id, {
            "password": new_password,
            "failed_login_attempts": 0,
            "locked_until": None
        }, updated_by=reset_by)
        
        # 撤销所有会话
        await self.revoke_all_sessions(user_id)
        
        # 记录操作日志
        self.log_operation("reset_password", user_id, {"reset_by": reset_by})
        
        return True
    
    async def lock_user(
        self,
        user_id: int,
        lock_duration: Optional[timedelta] = None,
        reason: Optional[str] = None,
        locked_by: Optional[int] = None
    ) -> bool:
        """
        锁定用户账户
        
        Args:
            user_id: 用户ID
            lock_duration: 锁定时长
            reason: 锁定原因
            locked_by: 锁定者ID
            
        Returns:
            是否锁定成功
        """
        lock_until = None
        if lock_duration:
            lock_until = datetime.utcnow() + lock_duration
        
        await self.update_user(user_id, {
            "locked_until": lock_until,
            "status": UserStatus.LOCKED
        }, updated_by=locked_by)
        
        # 撤销所有会话
        await self.revoke_all_sessions(user_id)
        
        # 记录操作日志
        self.log_operation("lock_user", user_id, {
            "reason": reason,
            "lock_until": lock_until.isoformat() if lock_until else None,
            "locked_by": locked_by
        })
        
        return True
    
    async def unlock_user(
        self,
        user_id: int,
        unlocked_by: Optional[int] = None
    ) -> bool:
        """
        解锁用户账户
        
        Args:
            user_id: 用户ID
            unlocked_by: 解锁者ID
            
        Returns:
            是否解锁成功
        """
        await self.update_user(user_id, {
            "locked_until": None,
            "failed_login_attempts": 0,
            "status": UserStatus.ACTIVE
        }, updated_by=unlocked_by)
        
        # 记录操作日志
        self.log_operation("unlock_user", user_id, {"unlocked_by": unlocked_by})
        
        return True
    
    async def activate_user(
        self,
        user_id: int,
        activated_by: Optional[int] = None
    ) -> bool:
        """
        激活用户账户
        
        Args:
            user_id: 用户ID
            activated_by: 激活者ID
            
        Returns:
            是否激活成功
        """
        await self.update_user(user_id, {
            "is_active": True,
            "status": UserStatus.ACTIVE,
            "email_verified_at": datetime.utcnow()
        }, updated_by=activated_by)
        
        # 记录操作日志
        self.log_operation("activate_user", user_id, {"activated_by": activated_by})
        
        return True
    
    async def deactivate_user(
        self,
        user_id: int,
        reason: Optional[str] = None,
        deactivated_by: Optional[int] = None
    ) -> bool:
        """
        停用用户账户
        
        Args:
            user_id: 用户ID
            reason: 停用原因
            deactivated_by: 停用者ID
            
        Returns:
            是否停用成功
        """
        await self.update_user(user_id, {
            "is_active": False,
            "status": UserStatus.INACTIVE
        }, updated_by=deactivated_by)
        
        # 撤销所有会话
        await self.revoke_all_sessions(user_id)
        
        # 记录操作日志
        self.log_operation("deactivate_user", user_id, {
            "reason": reason,
            "deactivated_by": deactivated_by
        })
        
        return True
    
    # =========================================================================
    # 用户角色管理
    # =========================================================================
    
    async def get_user_roles(self, user_id: int) -> List[UserRole]:
        """
        获取用户角色
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户角色列表
        """
        try:
            query = (
                select(UserRole)
                .join(user_roles)
                .where(user_roles.c.user_id == user_id)
                .where(UserRole.is_active == True)
            )
            
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            self.logger.error(f"Error getting user roles for user {user_id}: {e}")
            raise
    
    async def assign_role(
        self,
        user_id: int,
        role_id: int,
        assigned_by: Optional[int] = None,
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
        try:
            # 检查用户是否存在
            user = await self.get_by_id(user_id)
            if not user:
                raise ResourceNotFoundException("用户不存在")
            
            # 检查角色是否存在
            role_query = select(UserRole).where(UserRole.id == role_id)
            role_result = await self.db.execute(role_query)
            role = role_result.scalar_one_or_none()
            if not role:
                raise ResourceNotFoundException("角色不存在")
            
            # 检查是否已分配
            existing_query = (
                select(user_roles)
                .where(user_roles.c.user_id == user_id)
                .where(user_roles.c.role_id == role_id)
            )
            existing_result = await self.db.execute(existing_query)
            if existing_result.first():
                raise ConflictException("用户已拥有该角色")
            
            # 分配角色
            insert_stmt = user_roles.insert().values(
                user_id=user_id,
                role_id=role_id,
                assigned_by=assigned_by,
                assigned_at=datetime.utcnow(),
                expires_at=expires_at
            )
            
            await self.db.execute(insert_stmt)
            await self.db.commit()
            
            # 清理缓存
            await self.clear_user_cache(user_id)
            
            # 记录操作日志
            self.log_operation("assign_role", user_id, {
                "role_id": role_id,
                "assigned_by": assigned_by,
                "expires_at": expires_at.isoformat() if expires_at else None
            })
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error assigning role {role_id} to user {user_id}: {e}")
            raise
    
    async def remove_role(
        self,
        user_id: int,
        role_id: int,
        removed_by: Optional[int] = None
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
        try:
            # 删除角色分配
            delete_stmt = (
                delete(user_roles)
                .where(user_roles.c.user_id == user_id)
                .where(user_roles.c.role_id == role_id)
            )
            
            result = await self.db.execute(delete_stmt)
            if result.rowcount == 0:
                raise ResourceNotFoundException("用户角色分配不存在")
            
            await self.db.commit()
            
            # 清理缓存
            await self.clear_user_cache(user_id)
            
            # 记录操作日志
            self.log_operation("remove_role", user_id, {
                "role_id": role_id,
                "removed_by": removed_by
            })
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error removing role {role_id} from user {user_id}: {e}")
            raise
    
    async def has_role(self, user_id: int, role_name: str) -> bool:
        """
        检查用户是否拥有指定角色
        
        Args:
            user_id: 用户ID
            role_name: 角色名称
            
        Returns:
            是否拥有角色
        """
        try:
            query = (
                select(UserRole.id)
                .join(user_roles)
                .where(user_roles.c.user_id == user_id)
                .where(UserRole.name == role_name)
                .where(UserRole.is_active == True)
                .limit(1)
            )
            
            result = await self.db.execute(query)
            return result.scalar_one_or_none() is not None
            
        except Exception as e:
            self.logger.error(f"Error checking role {role_name} for user {user_id}: {e}")
            return False
    
    # =========================================================================
    # 用户权限管理
    # =========================================================================
    
    async def get_user_permissions(self, user_id: int) -> List[UserPermission]:
        """
        获取用户所有权限（包括角色权限和直接权限）
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户权限列表
        """
        try:
            # 获取角色权限
            role_permissions_query = (
                select(UserPermission)
                .join(role_permissions)
                .join(UserRole, UserRole.id == role_permissions.c.role_id)
                .join(user_roles, user_roles.c.role_id == UserRole.id)
                .where(user_roles.c.user_id == user_id)
                .where(UserRole.is_active == True)
                .where(UserPermission.is_active == True)
            )
            
            # 获取直接权限
            direct_permissions_query = (
                select(UserPermission)
                .join(user_permissions)
                .where(user_permissions.c.user_id == user_id)
                .where(UserPermission.is_active == True)
            )
            
            # 合并查询结果
            role_result = await self.db.execute(role_permissions_query)
            direct_result = await self.db.execute(direct_permissions_query)
            
            role_perms = role_result.scalars().all()
            direct_perms = direct_result.scalars().all()
            
            # 去重
            all_permissions = {}
            for perm in role_perms + direct_perms:
                all_permissions[perm.id] = perm
            
            return list(all_permissions.values())
            
        except Exception as e:
            self.logger.error(f"Error getting user permissions for user {user_id}: {e}")
            raise
    
    async def has_permission(self, user_id: int, permission_name: str) -> bool:
        """
        检查用户是否拥有指定权限
        
        Args:
            user_id: 用户ID
            permission_name: 权限名称
            
        Returns:
            是否拥有权限
        """
        try:
            # 检查超级用户
            user = await self.get_by_id(user_id)
            if user and user.is_superuser:
                return True
            
            # 检查角色权限
            role_permission_query = (
                select(UserPermission.id)
                .join(role_permissions)
                .join(UserRole, UserRole.id == role_permissions.c.role_id)
                .join(user_roles, user_roles.c.role_id == UserRole.id)
                .where(user_roles.c.user_id == user_id)
                .where(UserPermission.name == permission_name)
                .where(UserRole.is_active == True)
                .where(UserPermission.is_active == True)
                .limit(1)
            )
            
            result = await self.db.execute(role_permission_query)
            if result.scalar_one_or_none():
                return True
            
            # 检查直接权限
            direct_permission_query = (
                select(UserPermission.id)
                .join(user_permissions)
                .where(user_permissions.c.user_id == user_id)
                .where(UserPermission.name == permission_name)
                .where(UserPermission.is_active == True)
                .limit(1)
            )
            
            result = await self.db.execute(direct_permission_query)
            return result.scalar_one_or_none() is not None
            
        except Exception as e:
            self.logger.error(f"Error checking permission {permission_name} for user {user_id}: {e}")
            return False
    
    # =========================================================================
    # 用户会话管理
    # =========================================================================
    
    async def get_user_sessions(
        self,
        user_id: int,
        active_only: bool = True
    ) -> List[UserSession]:
        """
        获取用户会话
        
        Args:
            user_id: 用户ID
            active_only: 是否只获取活跃会话
            
        Returns:
            用户会话列表
        """
        try:
            query = select(UserSession).where(UserSession.user_id == user_id)
            
            if active_only:
                query = query.where(UserSession.status == SessionStatus.ACTIVE)
                query = query.where(UserSession.expires_at > datetime.utcnow())
            
            query = query.order_by(UserSession.last_activity_at.desc())
            
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            self.logger.error(f"Error getting user sessions for user {user_id}: {e}")
            raise
    
    async def revoke_session(
        self,
        session_id: int,
        revoked_by: Optional[int] = None
    ) -> bool:
        """
        撤销用户会话
        
        Args:
            session_id: 会话ID
            revoked_by: 撤销者ID
            
        Returns:
            是否撤销成功
        """
        try:
            update_stmt = (
                update(UserSession)
                .where(UserSession.id == session_id)
                .values(
                    status=SessionStatus.REVOKED,
                    revoked_at=datetime.utcnow(),
                    revoked_by=revoked_by
                )
            )
            
            result = await self.db.execute(update_stmt)
            if result.rowcount == 0:
                raise ResourceNotFoundException("会话不存在")
            
            await self.db.commit()
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error revoking session {session_id}: {e}")
            raise
    
    async def revoke_all_sessions(
        self,
        user_id: int,
        exclude_session_id: Optional[int] = None,
        revoked_by: Optional[int] = None
    ) -> int:
        """
        撤销用户所有会话
        
        Args:
            user_id: 用户ID
            exclude_session_id: 排除的会话ID
            revoked_by: 撤销者ID
            
        Returns:
            撤销的会话数量
        """
        try:
            query = (
                update(UserSession)
                .where(UserSession.user_id == user_id)
                .where(UserSession.status == SessionStatus.ACTIVE)
            )
            
            if exclude_session_id:
                query = query.where(UserSession.id != exclude_session_id)
            
            query = query.values(
                status=SessionStatus.REVOKED,
                revoked_at=datetime.utcnow(),
                revoked_by=revoked_by
            )
            
            result = await self.db.execute(query)
            await self.db.commit()
            
            # 记录操作日志
            self.log_operation("revoke_all_sessions", user_id, {
                "revoked_count": result.rowcount,
                "exclude_session_id": exclude_session_id,
                "revoked_by": revoked_by
            })
            
            return result.rowcount
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error revoking all sessions for user {user_id}: {e}")
            raise
    
    # =========================================================================
    # 用户统计和查询
    # =========================================================================
    
    async def get_user_statistics(self) -> Dict[str, Any]:
        """
        获取用户统计信息
        
        Returns:
            用户统计数据
        """
        try:
            # 总用户数
            total_users = await self.count()
            
            # 活跃用户数
            active_users = await self.count([User.is_active == True])
            
            # 已验证用户数
            verified_users = await self.count([User.email_verified_at.isnot(None)])
            
            # 按状态统计
            status_stats = {}
            for status in UserStatus:
                count = await self.count([User.status == status])
                status_stats[status.value] = count
            
            # 最近注册用户数（7天内）
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_registrations = await self.count([
                User.created_at >= seven_days_ago
            ])
            
            # 最近活跃用户数（30天内）
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_active = await self.count([
                User.last_activity_at >= thirty_days_ago
            ])
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "verified_users": verified_users,
                "status_distribution": status_stats,
                "recent_registrations": recent_registrations,
                "recent_active_users": recent_active,
                "verification_rate": verified_users / max(total_users, 1),
                "activation_rate": active_users / max(total_users, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user statistics: {e}")
            raise
    
    async def search_users(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        size: int = 20
    ) -> tuple[List[User], int]:
        """
        搜索用户
        
        Args:
            query: 搜索关键词
            filters: 额外过滤条件
            page: 页码
            size: 每页数量
            
        Returns:
            (用户列表, 总数)
        """
        try:
            # 构建搜索条件
            search_conditions = []
            if query:
                search_conditions.append(
                    or_(
                        User.username.ilike(f"%{query}%"),
                        User.email.ilike(f"%{query}%"),
                        User.first_name.ilike(f"%{query}%"),
                        User.last_name.ilike(f"%{query}%"),
                        User.display_name.ilike(f"%{query}%")
                    )
                )
            
            # 添加额外过滤条件
            if filters:
                for field, value in filters.items():
                    if hasattr(User, field) and value is not None:
                        search_conditions.append(getattr(User, field) == value)
            
            # 执行分页查询
            return await self.get_with_pagination(
                page=page,
                size=size,
                filters=search_conditions,
                order_by="created_at",
                order_desc=True
            )
            
        except Exception as e:
            self.logger.error(f"Error searching users with query '{query}': {e}")
            raise
    
    # =========================================================================
    # 缓存管理
    # =========================================================================
    
    async def clear_user_cache(self, user_id: int):
        """
        清理用户相关缓存
        
        Args:
            user_id: 用户ID
        """
        cache_patterns = [
            f"user:{user_id}:*",
            f"user:permissions:{user_id}",
            f"user:roles:{user_id}",
            f"user:sessions:{user_id}"
        ]
        
        for pattern in cache_patterns:
            await self.cache.clear_pattern(pattern)
    
    # =========================================================================
    # 数据验证
    # =========================================================================
    
    def validate_user_data(
        self,
        data: Dict[str, Any],
        is_update: bool = False
    ) -> Dict[str, Any]:
        """
        验证用户数据
        
        Args:
            data: 用户数据
            is_update: 是否为更新操作
            
        Returns:
            验证后的数据
        """
        # 验证用户名
        if "username" in data:
            username = data["username"]
            if not username or len(username) < 3 or len(username) > 50:
                raise ValidationException("用户名长度必须在3-50个字符之间")
            if not username.replace("_", "").replace("-", "").isalnum():
                raise ValidationException("用户名只能包含字母、数字、下划线和连字符")
        
        # 验证邮箱
        if "email" in data:
            email = data["email"]
            if not email or "@" not in email:
                raise ValidationException("邮箱格式无效")
        
        # 验证手机号
        if "phone" in data and data["phone"]:
            phone = data["phone"]
            if not phone.isdigit() or len(phone) != 11:
                raise ValidationException("手机号格式无效")
        
        # 验证性别
        if "gender" in data and data["gender"]:
            if data["gender"] not in [g.value for g in UserGender]:
                raise ValidationException("性别值无效")
        
        # 验证生日
        if "birth_date" in data and data["birth_date"]:
            birth_date = data["birth_date"]
            if isinstance(birth_date, str):
                try:
                    birth_date = datetime.fromisoformat(birth_date).date()
                    data["birth_date"] = birth_date
                except ValueError:
                    raise ValidationException("生日格式无效")
            
            if birth_date > datetime.now().date():
                raise ValidationException("生日不能是未来日期")
        
        return data