#!/usr/bin/env python3
"""
用户模型模块

定义用户相关的数据模型，包括：
1. 用户模型
2. 用户角色模型
3. 用户权限模型
4. 用户会话模型
5. 用户配置模型
6. 用户关系模型
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey,
    Table, UniqueConstraint, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func

from .base import BaseModel, TimestampMixin, SoftDeleteMixin, AuditMixin, MetadataMixin
from ..core.security import SecurityManager


# =============================================================================
# 枚举定义
# =============================================================================

class UserStatus(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"          # 活跃
    INACTIVE = "inactive"      # 非活跃
    SUSPENDED = "suspended"    # 暂停
    BANNED = "banned"          # 封禁
    PENDING = "pending"        # 待激活


class UserGender(str, Enum):
    """用户性别枚举"""
    MALE = "male"              # 男性
    FEMALE = "female"          # 女性
    OTHER = "other"            # 其他
    UNKNOWN = "unknown"        # 未知


class RoleType(str, Enum):
    """角色类型枚举"""
    SYSTEM = "system"          # 系统角色
    CUSTOM = "custom"          # 自定义角色
    TEMPORARY = "temporary"    # 临时角色


class PermissionType(str, Enum):
    """权限类型枚举"""
    READ = "read"              # 读取权限
    WRITE = "write"            # 写入权限
    DELETE = "delete"          # 删除权限
    ADMIN = "admin"            # 管理权限
    EXECUTE = "execute"        # 执行权限


class SessionStatus(str, Enum):
    """会话状态枚举"""
    ACTIVE = "active"          # 活跃
    EXPIRED = "expired"        # 过期
    REVOKED = "revoked"        # 撤销
    LOGOUT = "logout"          # 登出


# =============================================================================
# 关联表定义
# =============================================================================

# 用户角色关联表
user_roles = Table(
    'user_roles',
    BaseModel.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('role_id', Integer, ForeignKey('user_roles_table.id', ondelete='CASCADE'), primary_key=True),
    Column('assigned_at', DateTime(timezone=True), server_default=func.now()),
    Column('assigned_by', Integer, ForeignKey('users.id')),
    Column('expires_at', DateTime(timezone=True), nullable=True),
    Index('idx_user_roles_user_id', 'user_id'),
    Index('idx_user_roles_role_id', 'role_id'),
)

# 角色权限关联表
role_permissions = Table(
    'role_permissions',
    BaseModel.metadata,
    Column('role_id', Integer, ForeignKey('user_roles_table.id', ondelete='CASCADE'), primary_key=True),
    Column('permission_id', Integer, ForeignKey('user_permissions.id', ondelete='CASCADE'), primary_key=True),
    Column('granted_at', DateTime(timezone=True), server_default=func.now()),
    Column('granted_by', Integer, ForeignKey('users.id')),
    Index('idx_role_permissions_role_id', 'role_id'),
    Index('idx_role_permissions_permission_id', 'permission_id'),
)

# 用户直接权限关联表
user_permissions = Table(
    'user_direct_permissions',
    BaseModel.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('permission_id', Integer, ForeignKey('user_permissions.id', ondelete='CASCADE'), primary_key=True),
    Column('granted_at', DateTime(timezone=True), server_default=func.now()),
    Column('granted_by', Integer, ForeignKey('users.id')),
    Column('expires_at', DateTime(timezone=True), nullable=True),
    Index('idx_user_permissions_user_id', 'user_id'),
    Index('idx_user_permissions_permission_id', 'permission_id'),
)


# =============================================================================
# 用户模型
# =============================================================================

class User(BaseModel, TimestampMixin, SoftDeleteMixin, AuditMixin, MetadataMixin):
    """用户模型"""
    
    __tablename__ = "users"
    
    # 基本信息
    username = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        comment="用户名"
    )
    
    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="邮箱地址"
    )
    
    phone = Column(
        String(20),
        unique=True,
        nullable=True,
        index=True,
        comment="手机号码"
    )
    
    password_hash = Column(
        String(255),
        nullable=False,
        comment="密码哈希"
    )
    
    # 个人信息
    first_name = Column(
        String(50),
        nullable=True,
        comment="名字"
    )
    
    last_name = Column(
        String(50),
        nullable=True,
        comment="姓氏"
    )
    
    display_name = Column(
        String(100),
        nullable=True,
        comment="显示名称"
    )
    
    avatar_url = Column(
        String(500),
        nullable=True,
        comment="头像URL"
    )
    
    gender = Column(
        SQLEnum(UserGender),
        default=UserGender.UNKNOWN,
        comment="性别"
    )
    
    birth_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="出生日期"
    )
    
    bio = Column(
        Text,
        nullable=True,
        comment="个人简介"
    )
    
    # 状态信息
    status = Column(
        SQLEnum(UserStatus),
        default=UserStatus.PENDING,
        nullable=False,
        index=True,
        comment="用户状态"
    )
    
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="是否激活"
    )
    
    is_superuser = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="是否超级用户"
    )
    
    is_staff = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="是否员工"
    )
    
    is_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="是否已验证"
    )
    
    # 时间信息
    last_login_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后登录时间"
    )
    
    last_activity_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后活动时间"
    )
    
    email_verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="邮箱验证时间"
    )
    
    phone_verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="手机验证时间"
    )
    
    password_changed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="密码修改时间"
    )
    
    # 安全信息
    failed_login_attempts = Column(
        Integer,
        default=0,
        nullable=False,
        comment="失败登录次数"
    )
    
    locked_until = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="锁定到期时间"
    )
    
    two_factor_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否启用双因子认证"
    )
    
    two_factor_secret = Column(
        String(255),
        nullable=True,
        comment="双因子认证密钥"
    )
    
    # 偏好设置
    timezone = Column(
        String(50),
        default="UTC",
        comment="时区"
    )
    
    language = Column(
        String(10),
        default="en",
        comment="语言"
    )
    
    theme = Column(
        String(20),
        default="light",
        comment="主题"
    )
    
    # 关系定义
    roles = relationship(
        "UserRole",
        secondary=user_roles,
        back_populates="users",
        lazy="selectin"
    )
    
    direct_permissions = relationship(
        "UserPermission",
        secondary=user_permissions,
        back_populates="users",
        lazy="selectin"
    )
    
    sessions = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_users_email_status', 'email', 'status'),
        Index('idx_users_username_status', 'username', 'status'),
        Index('idx_users_last_activity', 'last_activity_at'),
        Index('idx_users_created_at', 'created_at'),
    )
    
    # 混合属性
    @hybrid_property
    def full_name(self) -> str:
        """完整姓名"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.username
    
    @hybrid_property
    def is_locked(self) -> bool:
        """是否被锁定"""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until
    
    @hybrid_property
    def is_email_verified(self) -> bool:
        """邮箱是否已验证"""
        return self.email_verified_at is not None
    
    @hybrid_property
    def is_phone_verified(self) -> bool:
        """手机是否已验证"""
        return self.phone_verified_at is not None
    
    # 方法定义
    def set_password(self, password: str) -> None:
        """
        设置密码
        
        Args:
            password: 明文密码
        """
        security_manager = SecurityManager()
        self.password_hash = security_manager.hash_password(password)
        self.password_changed_at = datetime.utcnow()
    
    def verify_password(self, password: str) -> bool:
        """
        验证密码
        
        Args:
            password: 明文密码
        
        Returns:
            bool: 密码是否正确
        """
        security_manager = SecurityManager()
        return security_manager.verify_password(password, self.password_hash)
    
    def has_role(self, role_name: str) -> bool:
        """
        检查是否有指定角色
        
        Args:
            role_name: 角色名称
        
        Returns:
            bool: 是否有该角色
        """
        if self.is_superuser:
            return True
        
        return any(role.name == role_name for role in self.roles)
    
    def has_permission(self, permission_name: str) -> bool:
        """
        检查是否有指定权限
        
        Args:
            permission_name: 权限名称
        
        Returns:
            bool: 是否有该权限
        """
        if self.is_superuser:
            return True
        
        # 检查直接权限
        if any(perm.name == permission_name for perm in self.direct_permissions):
            return True
        
        # 检查角色权限
        for role in self.roles:
            if role.has_permission(permission_name):
                return True
        
        return False
    
    def get_all_permissions(self) -> List[str]:
        """
        获取所有权限
        
        Returns:
            List[str]: 权限名称列表
        """
        if self.is_superuser:
            return ["*"]  # 超级用户拥有所有权限
        
        permissions = set()
        
        # 添加直接权限
        for perm in self.direct_permissions:
            permissions.add(perm.name)
        
        # 添加角色权限
        for role in self.roles:
            for perm in role.permissions:
                permissions.add(perm.name)
        
        return list(permissions)
    
    def update_last_login(self) -> None:
        """更新最后登录时间"""
        self.last_login_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()
    
    def update_last_activity(self) -> None:
        """更新最后活动时间"""
        self.last_activity_at = datetime.utcnow()
    
    def lock_account(self, duration_minutes: int = 30) -> None:
        """
        锁定账户
        
        Args:
            duration_minutes: 锁定时长（分钟）
        """
        self.locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
    
    def unlock_account(self) -> None:
        """解锁账户"""
        self.locked_until = None
        self.failed_login_attempts = 0
    
    def increment_failed_login(self) -> None:
        """增加失败登录次数"""
        self.failed_login_attempts += 1
    
    def reset_failed_login(self) -> None:
        """重置失败登录次数"""
        self.failed_login_attempts = 0
    
    def verify_email(self) -> None:
        """验证邮箱"""
        self.email_verified_at = datetime.utcnow()
        self.is_verified = True
        if self.status == UserStatus.PENDING:
            self.status = UserStatus.ACTIVE
    
    def verify_phone(self) -> None:
        """验证手机"""
        self.phone_verified_at = datetime.utcnow()
    
    def enable_two_factor(self, secret: str) -> None:
        """
        启用双因子认证
        
        Args:
            secret: 双因子认证密钥
        """
        self.two_factor_enabled = True
        self.two_factor_secret = secret
    
    def disable_two_factor(self) -> None:
        """禁用双因子认证"""
        self.two_factor_enabled = False
        self.two_factor_secret = None
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        转换为字典
        
        Args:
            include_sensitive: 是否包含敏感信息
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        exclude = ['password_hash']
        if not include_sensitive:
            exclude.extend([
                'two_factor_secret',
                'failed_login_attempts',
                'locked_until'
            ])
        
        data = super().to_dict(exclude=exclude)
        
        # 添加计算属性
        data.update({
            'full_name': self.full_name,
            'is_locked': self.is_locked,
            'is_email_verified': self.is_email_verified,
            'is_phone_verified': self.is_phone_verified,
            'roles': [role.name for role in self.roles],
            'permissions': self.get_all_permissions(),
        })
        
        return data


# =============================================================================
# 用户角色模型
# =============================================================================

class UserRole(BaseModel, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """用户角色模型"""
    
    __tablename__ = "user_roles_table"
    
    name = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        comment="角色名称"
    )
    
    display_name = Column(
        String(100),
        nullable=True,
        comment="显示名称"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="角色描述"
    )
    
    role_type = Column(
        SQLEnum(RoleType),
        default=RoleType.CUSTOM,
        nullable=False,
        comment="角色类型"
    )
    
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="是否激活"
    )
    
    priority = Column(
        Integer,
        default=0,
        nullable=False,
        comment="优先级（数字越大优先级越高）"
    )
    
    # 关系定义
    users = relationship(
        "User",
        secondary=user_roles,
        back_populates="roles",
        lazy="dynamic"
    )
    
    permissions = relationship(
        "UserPermission",
        secondary=role_permissions,
        back_populates="roles",
        lazy="selectin"
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_user_roles_name_type', 'name', 'role_type'),
        Index('idx_user_roles_priority', 'priority'),
    )
    
    def has_permission(self, permission_name: str) -> bool:
        """
        检查角色是否有指定权限
        
        Args:
            permission_name: 权限名称
        
        Returns:
            bool: 是否有该权限
        """
        return any(perm.name == permission_name for perm in self.permissions)
    
    def add_permission(self, permission: "UserPermission") -> None:
        """
        添加权限
        
        Args:
            permission: 权限对象
        """
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: "UserPermission") -> None:
        """
        移除权限
        
        Args:
            permission: 权限对象
        """
        if permission in self.permissions:
            self.permissions.remove(permission)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        data = super().to_dict()
        data['permissions'] = [perm.name for perm in self.permissions]
        return data


# =============================================================================
# 用户权限模型
# =============================================================================

class UserPermission(BaseModel, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """用户权限模型"""
    
    __tablename__ = "user_permissions"
    
    name = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="权限名称"
    )
    
    display_name = Column(
        String(100),
        nullable=True,
        comment="显示名称"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="权限描述"
    )
    
    permission_type = Column(
        SQLEnum(PermissionType),
        default=PermissionType.READ,
        nullable=False,
        comment="权限类型"
    )
    
    resource = Column(
        String(100),
        nullable=True,
        comment="资源名称"
    )
    
    action = Column(
        String(50),
        nullable=True,
        comment="操作名称"
    )
    
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="是否激活"
    )
    
    # 关系定义
    roles = relationship(
        "UserRole",
        secondary=role_permissions,
        back_populates="permissions",
        lazy="dynamic"
    )
    
    users = relationship(
        "User",
        secondary=user_permissions,
        back_populates="direct_permissions",
        lazy="dynamic"
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_user_permissions_name_type', 'name', 'permission_type'),
        Index('idx_user_permissions_resource_action', 'resource', 'action'),
        UniqueConstraint('resource', 'action', name='uq_permission_resource_action'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        return super().to_dict()


# =============================================================================
# 用户会话模型
# =============================================================================

class UserSession(BaseModel, TimestampMixin):
    """用户会话模型"""
    
    __tablename__ = "user_sessions"
    
    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    session_token = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="会话令牌"
    )
    
    refresh_token = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        comment="刷新令牌"
    )
    
    status = Column(
        SQLEnum(SessionStatus),
        default=SessionStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="会话状态"
    )
    
    ip_address = Column(
        String(45),
        nullable=True,
        comment="IP地址"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        comment="用户代理"
    )
    
    device_info = Column(
        Text,
        nullable=True,
        comment="设备信息"
    )
    
    location = Column(
        String(200),
        nullable=True,
        comment="地理位置"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="过期时间"
    )
    
    last_activity_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="最后活动时间"
    )
    
    revoked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="撤销时间"
    )
    
    revoked_by = Column(
        Integer,
        ForeignKey('users.id'),
        nullable=True,
        comment="撤销者ID"
    )
    
    # 关系定义
    user = relationship(
        "User",
        back_populates="sessions",
        foreign_keys=[user_id]
    )
    
    revoker = relationship(
        "User",
        foreign_keys=[revoked_by]
    )
    
    # 索引定义
    __table_args__ = (
        Index('idx_user_sessions_user_status', 'user_id', 'status'),
        Index('idx_user_sessions_expires_at', 'expires_at'),
        Index('idx_user_sessions_ip_address', 'ip_address'),
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """是否已过期"""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_active(self) -> bool:
        """是否活跃"""
        return (
            self.status == SessionStatus.ACTIVE and
            not self.is_expired
        )
    
    def revoke(self, revoked_by_user_id: Optional[int] = None) -> None:
        """
        撤销会话
        
        Args:
            revoked_by_user_id: 撤销者用户ID
        """
        self.status = SessionStatus.REVOKED
        self.revoked_at = datetime.utcnow()
        if revoked_by_user_id:
            self.revoked_by = revoked_by_user_id
    
    def extend_expiry(self, hours: int = 24) -> None:
        """
        延长过期时间
        
        Args:
            hours: 延长的小时数
        """
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
    
    def update_activity(self) -> None:
        """更新活动时间"""
        self.last_activity_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        data = super().to_dict(exclude=['session_token', 'refresh_token'])
        data.update({
            'is_expired': self.is_expired,
            'is_active': self.is_active,
        })
        return data